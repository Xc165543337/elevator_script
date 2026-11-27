"""Boss-farming policy driven by the solved MDP (mdp.py).

MarkovPolicy keeps everything BossFarmPolicy does around a fight (events, temp gear, safe box,
collectibles, adrenaline on F1) and replaces the threshold rules for the decisions that spend the run's
resources: heal / fight / flee in encounters and boss windows, and heal / step / rush between floors.

Each decision:
  1. picks the heal price lambda for the live heal stock (dual bound of the relaxed budget);
  2. evaluates the legal actions under that price - fights are simulated live against the actual enemy
     with the live build, the continuation is read from the solved tables;
  3. takes the best one. Rushing is preferred over single steps while it gives up at most
     `rush_tolerance` expected bosses (single steps are never worse, only slower).
Below `from_floor`, outside the solved horizon and against extraction ambushers it defers to
BossFarmPolicy.
"""
from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass, field, fields, replace
from typing import Callable

import numpy as np

import field as field_items
from combat import Calibration, battle_enemy
from field import FieldPrep
from mdp import (BOSS_WINDOWS, FIELD_HEALS, SHOP_HEALS, HealItem, MdpConfig, ValueTables, _Closure,
                 elite_pause_kernel, elite_pause_value, flee_chance, flee_hit, interp_matrix, load_tables,
                 save_tables, solve, tables_key, with_calibration)
from model import (BOSS_WINDOW, Build, boss_win_chance, fight, fight_segment, foe_at, next_boss_floors,
                   pauses_after_segment)
from policy import KEEP_IN_BAG, BossFarmPolicy, Decision, PolicyConfig

WISH = "field_wish"   # 为了我所期待的


@dataclass
class MarkovConfig:
    from_floor: int = 30            # shallower floors are trivial and the build is still ramping up
    # expected bosses a rush may give up against single steps; it is paid on every rush of the run, so it
    # only absorbs Monte Carlo noise (0.02 let a two-fight rush at 3.6% death risk through at F1205 live)
    rush_tolerance: float = 0.002
    # expected bosses a rush with 自动打精英 may give up against the best other move (fewer stops per run)
    elite_rush_tolerance: float = 0.002
    heal_margin: float = 0.002      # a heal / flee must beat the alternative by this many bosses
    prep_margin: float = 0.01       # a field buff / grenade (no budget price) must gain this many bosses
    prep_pairs: int = 3             # best single preparations combined pairwise (buff + buff synergies)
    prep_min_death: float = 0.001   # search field preparations only against regular fights at least this deadly
    # Extract once the expected further boss kills drop below this (0 = never): the run has hit a wall, and
    # extracting (or a smoke bomb against a boss, which also ends the run as an extraction) banks the bag that
    # a death would lose. Estimated optimistically (see _boss_outlook), so it only fires at a real wall.
    extract_below: float = 0.3
    outlook_bosses: int = 3         # bosses ahead that the expected-kills estimate chains
    live_samples: int = 300         # Monte Carlo fights per HP grid point for live evaluations
    pause_samples: int = 48         # live fights per (hp, elite hp) point when valuing the elite pause
    # 为了我所期待的 (one per run at F500, 0.1% per floor): kills the current enemy, wipes this run's loot.
    # Only for a real emergency: the coming fight segment must stay at least wish_min_death deadly even
    # after the best heal at hand (a boss that cannot kill from full HP in one window is healed through,
    # not wished away), and the wish must add wish_gain of the value at stake over the best alternative.
    wish_gain: float = 0.15
    wish_min_death: float = 0.05
    wish_max_loot: int = 15_000_000    # never wipe more run loot (bag + safe box) than this with a wish
    build_tolerance: float = 0.10   # relative stat drift before re-solving (the live fight always uses the exact build)
    # Tables are solved for the build snapped down to a geometric grid with this step (bucket()), so that
    # similar builds of different runs share one cached solve. The live fight still uses the exact build.
    bucket: float = 0.05
    background: bool = True         # solve in a thread and keep playing on the previous tables meanwhile
    cache_dir: str = "cache"
    cache_max_files: int = 20       # least recently used solves beyond this are deleted (~55 MB each)
    mdp: MdpConfig = field(default_factory=MdpConfig)


@dataclass(frozen=True)
class HealOption:
    item: HealItem
    decision: Decision


_HEAL_TEXT = re.compile(r"回复 (\d+)% 最大生命")


def _field_heal_spec(item: dict) -> HealItem | None:
    """Instant field item (heal and / or shield) as a HealItem; None for everything else."""
    raw = field_items.SPECS.get(item.get("id", ""))
    if raw is not None:
        if raw["mode"] != "instant" or raw.get("cost_hp") or not (raw.get("heal") or raw.get("shield")):
            return None
        return HealItem(raw["name"], heal=raw.get("heal", 0.0), shield_add=raw.get("shield", 0.0))
    spec = FIELD_HEALS.get(item.get("name", ""))
    if spec is None and item.get("healing"):
        m = _HEAL_TEXT.search(item.get("description", ""))
        spec = HealItem(item.get("name", "?"), heal=int(m.group(1)) / 100) if m else None
    return spec


def heal_stock(run: dict) -> float:
    """Heal budget in max-HP units: carried medicine plus instant field heals."""
    max_hp = float(run["max_hp"])
    total = 0.0
    for it in run.get("bag") or []:
        spec = SHOP_HEALS.get(it.get("id"))
        if spec:
            total += spec.units(max_hp) * (it.get("quantity") or 1)
    for it in run.get("field") or []:
        spec = _field_heal_spec(it)
        if spec:
            total += spec.units(max_hp) * (it.get("quantity") or 0)
    return total


def entry_heal_pending(run: dict) -> bool:
    """HP rises on the next floor entry (slow serum regen, field regen). The server's rush stop rule
    (HP <= 50%) reads HP after that tick, which the tables do not model - so a rush started low could
    auto-fight a deep monster at barely half HP."""
    recovery = run.get("recovery") or {}
    return bool(recovery.get("left")) or field_items.active_mods(run).get("regen", 0.0) > 0


def smoke_count(run: dict) -> int:
    return sum(it.get("quantity") or 1 for it in run.get("bag") or [] if it.get("id") == "smoke")


def bag_at_risk(run: dict) -> float:
    """What a death would lose now, in million gold: the bag (the safe box is kept)."""
    return sum(float(it.get("value") or 0) * (it.get("quantity") or 1) for it in run.get("bag") or []) / 1e6


def effective_hp(run: dict) -> float:
    return (run["hp"] + (run.get("shield") or 0)) / run["max_hp"]


def isotonic(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool adjacent violators). Values of a state can only grow
    with HP, so this removes Monte Carlo noise between neighbouring grid points without bias."""
    blocks: list[list[float]] = []          # [mean, size]
    for v in map(float, y):
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([np.full(int(n), v) for v, n in blocks])


class Planner:
    """Value tables for the current build: in memory, then the disk cache, then a solve.

    A solve takes minutes, so by default it runs in a background thread: until it is done, tables()
    returns the previous tables of this map, or None (the policy then plays by its threshold rules).
    The live fight is always simulated with the exact build either way."""

    def __init__(self, cfg: MarkovConfig, log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.log = log or (lambda _msg: None)
        self._tables: ValueTables | None = None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._wanted: tuple[Build, str] | None = None   # latest solve asked for and not yet done

    def tables(self, build: Build, map_id: str) -> ValueTables | None:
        current = self._tables
        if current is not None and current.map_id == map_id and self._close(current.build, build):
            return current
        target = bucket(quantize(build), self.cfg.bucket)
        cached = self._load(target, map_id)
        if cached is not None:
            self._tables = cached
            return cached
        if not self.cfg.background:
            self._tables = self._solve(target, map_id)
            return self._tables
        self._request(target, map_id)
        return current if current is not None and current.map_id == map_id else None

    def wait(self) -> None:
        """Block until a background solve (if any) is done."""
        worker = self._worker
        if worker is not None:
            worker.join()

    def _path(self, build: Build, map_id: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"mdp_{tables_key(build, map_id, self.cfg.mdp)}.npz")

    def _load(self, build: Build, map_id: str) -> ValueTables | None:
        path = self._path(build, map_id)
        if not os.path.exists(path):
            return None
        os.utime(path)   # most recently used: kept longest by _prune
        return load_tables(path, build, map_id, self.cfg.mdp)

    def _solve(self, build: Build, map_id: str) -> ValueTables:
        self.log(f"mdp: solving tables for {map_id} (atk {build.atk:.0f}, def {build.defense:.0f}, "
                 f"agi {build.agi:.0f}, hp {build.hp:.0f})")
        tables = solve(build, map_id, self.cfg.mdp, progress=self.log)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = self._path(build, map_id)
        save_tables(tables, path)
        self.log(f"mdp: solved in {tables.solve_seconds:.0f}s -> {path}")
        self._prune()
        return tables

    def _request(self, build: Build, map_id: str) -> None:
        with self._lock:
            self._wanted = (build, map_id)
            if self._worker is not None and self._worker.is_alive():
                return              # the worker picks up the latest request when its solve is done
            self._worker = threading.Thread(target=self._work, name="mdp-solve", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                wanted, self._wanted = self._wanted, None
            if wanted is None:
                return
            build, map_id = wanted
            try:
                tables = self._load(build, map_id) or self._solve(build, map_id)
            except Exception as err:   # never take the farm down with it: keep the old tables
                self.log(f"mdp: background solve failed: {err!r}")
                return
            self._tables = tables

    def _prune(self) -> None:
        """Keep the cache_max_files most recently used solves."""
        try:
            files = [os.path.join(self.cfg.cache_dir, n) for n in os.listdir(self.cfg.cache_dir)
                     if n.startswith("mdp_") and n.endswith(".npz")]
        except FileNotFoundError:
            return
        files.sort(key=os.path.getmtime, reverse=True)
        for path in files[self.cfg.cache_max_files:]:
            os.remove(path)
            self.log(f"mdp: removed old cached tables {os.path.basename(path)}")

    def _close(self, a: Build, b: Build) -> bool:
        """Same build up to `build_tolerance` relative drift in any number (e.g. the online calibration)."""
        tol = self.cfg.build_tolerance
        for f in fields(Build):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                if f.name != "label" and x != y:
                    return False
            elif abs(x - y) > tol * max(abs(x), abs(y), 1.0):
                return False
        return True


# Higher is worse for these: rounded up by bucket(), everything else down.
_WORSE_WHEN_HIGHER = frozenset({"taken_mult"})
_WHOLE = frozenset({"hp", "atk", "defense", "agi", "luk"})   # whole numbers in the panel


def _snap(x: float, step: float, up: bool, whole: bool) -> float:
    """The grid point (1 + step)^k nearest to x from below (from above when `up`). Grid points of whole
    stats are rounded, and a rounded grid point maps to itself."""
    ratio = math.log1p(step)

    def point(k: int) -> float:
        v = math.exp(k * ratio)
        return round(v) if whole else v

    k = math.floor(math.log(x) / ratio)
    if up:
        while point(k) < x:
            k += 1
        while point(k - 1) >= x:
            k -= 1
    else:
        while point(k + 1) <= x:
            k += 1
        while point(k) > x:
            k -= 1
    return point(k)


def bucket(build: Build, step: float) -> Build:
    """Snap every positive number to the grid (1 + step)^k in the conservative direction (down, or up
    for damage taken), so that builds within a few percent of each other share cached tables."""
    if step <= 0:
        return build
    changes = {}
    for f in fields(Build):
        x = getattr(build, f.name)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0:
            continue
        changes[f.name] = _snap(float(x), step, f.name in _WORSE_WHEN_HIGHER, f.name in _WHOLE)
        if f.name in _WHOLE:
            changes[f.name] = int(changes[f.name])
    return replace(build, **changes)


def quantize(build: Build) -> Build:
    """Round a build for solving and caching: whole stats, two-decimal damage multipliers rounded in
    the conservative direction (less damage dealt, more taken)."""
    return replace(build, hp=round(build.hp), atk=round(build.atk), defense=round(build.defense),
                   agi=round(build.agi), luk=round(build.luk),
                   dealt_mult=math.floor(build.dealt_mult * 100) / 100,
                   taken_mult=math.ceil(build.taken_mult * 100) / 100)


class MarkovPolicy(BossFarmPolicy):
    def __init__(self, cal: Calibration, config: PolicyConfig | None = None,
                 markov: MarkovConfig | None = None, log: Callable[[str], None] | None = None):
        super().__init__(cal, config)
        markov = markov or MarkovConfig()
        # one source of truth for the server's round limits and rush options: the policy config
        self.mcfg = replace(markov, mdp=replace(markov.mdp, rules=self.cfg.rules, rush_elites=self.cfg.rush_elites))
        self.planner = Planner(self.mcfg, log)
        self._rng = np.random.default_rng()
        self._wish_sent = False   # a wish preview was requested; confirm it on the next decision
        self._boss_odds: dict[tuple, float] = {}   # (build, map, boss floor) -> boss_win_chance

    # ------------------------------------------------------------------ helpers
    def _calibrated(self, build: Build) -> Build:
        kp, ke = self.cal.conservative()
        return with_calibration(build, kp, ke)

    def _tables(self, state: dict) -> ValueTables | None:
        run = state["run"]
        if run["floor"] < self.mcfg.from_floor:
            return None
        tables = self.planner.tables(self._calibrated(self._table_build(state)), run.get("map") or "mall")
        return tables if tables is not None and tables.in_range(run["floor"]) else None

    def _table_build(self, state: dict) -> Build:
        """Build the tables are solved for: lasting_build without short boosts it cannot see as buffs,
        i.e. agility capped at panel + worn gear (ice tea gives +15 for two floors only)."""
        build = self.lasting_build(state)
        run = state["run"]
        base = (run.get("stats") or {}).get("agi")
        if base is None:
            return build
        lasting = max(0.0, base + field_items.worn_mods(run).get("agi", 0.0))
        return replace(build, agi=min(build.agi, lasting))

    def _heal_options(self, run: dict, in_combat: bool) -> list[HealOption]:
        floor, out, seen = run["floor"], [], set()
        for it in run.get("bag") or []:
            spec = SHOP_HEALS.get(it.get("id"))
            if spec is None or spec.id in seen or (spec.combat_only and not in_combat):
                continue
            params = {"index": it["index"]}
            if self._item_blocked(run, spec.id) or not self._allowed(floor, "use", params):
                continue
            seen.add(spec.id)
            out.append(HealOption(spec, Decision("use", params, f"use {it.get('name', spec.id)}")))
        for it in run.get("field") or []:
            spec = _field_heal_spec(it)
            params = {"id": it.get("id")}
            if spec is None or (it.get("quantity") or 0) <= 0 or spec.id in seen \
                    or not self._allowed(floor, "field_use", params):
                continue
            seen.add(spec.id)
            out.append(HealOption(spec, Decision("field_use", params, f"field heal {it.get('name')}")))
        return out

    def _choose(self, e: float, lam: float, max_hp: float, curve: np.ndarray, grid: np.ndarray,
                options: list[HealOption]) -> tuple[HealOption | None, float, float]:
        """Best heal under price lam against continuing at `e`.
        curve: value of the best non-heal action on the grid. Returns (option or None, Q(best heal), Q(none))."""
        closure = _Closure([o.item for o in options], grid, max_hp, np.array([lam]), self.mcfg.mdp.heal_iterations)
        healed = closure(curve[None, :])[0]
        q_none = float(np.interp(e, grid, curve))
        best, q_best = None, -np.inf
        for o in options:
            target = float(o.item.apply(np.array([e]), max_hp)[0])
            if target <= e + 1e-9:
                continue
            q = float(np.interp(target, grid, healed)) - lam * o.item.units(max_hp)
            if q > q_best:
                best, q_best = o, q
        return best, q_best, q_none

    def _sim_starts(self, grid: np.ndarray, build: Build) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, s = len(grid), self.mcfg.live_samples
        start = np.repeat(np.arange(1, n), s)
        e0 = grid[start]
        return start, np.minimum(e0, 1.0) * build.hp, np.maximum(e0 - 1.0, 0.0) * build.hp

    def _per_point(self, start: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
        """Mean of `values` per start grid point (point 0 = dead = 0)."""
        out = np.bincount(start, weights=values, minlength=n) / self.mcfg.live_samples
        out[0] = 0.0
        return out

    def _live_build(self, state: dict) -> Build:
        folded = (state["player"].get("secondary") or {}).get("mitigation") is not None
        return self._calibrated(field_items.live_build(state["run"], self.current_build(state), folded))

    # ------------------------------------------------------------------ encounters
    def _fight(self, state: dict, ctr: dict) -> Decision:
        run = state["run"]
        enemy = run["event"]["enemy"]
        tables = None if enemy.get("ambush") else self._tables(state)
        if tables is None:
            return super()._fight(state, ctr)
        return self._encounter(state, ctr, tables)

    def _fight_curve(self, state: dict, tables: ValueTables, li: int, build: Build,
                     enemy: dict) -> tuple[np.ndarray, np.ndarray]:
        """(value, death probability) of fighting now, for every effective-HP grid point.
        Regular enemies: the fight up to its cap, then the ready table; an elite under live rules only
        up to its pause, valued with heals at the pause and the remaining rounds. Bosses: one window,
        then the boss table. The death probability covers this segment only."""
        g, grid = state["run"]["floor"], tables.grid
        cost = bag_at_risk(state["run"]) if tables.config.gold else 0.0   # gold objective: a death loses the bag
        start, hp, shield = self._sim_starts(grid, build)
        ehp = np.full(len(start), float(enemy["hp"]))
        if enemy.get("boss"):
            fought = int(enemy.get("rounds_fought") or 0)
            window = fought // BOSS_WINDOW + 1
            boss_max = float(enemy.get("max_hp") or enemy["hp"])
            hp, shield, ehp, *_ = fight(build, hp, shield, enemy, boss=True, ehp=ehp, start_round=fought + 1,
                                        rounds=BOSS_WINDOW - fought % BOSS_WINDOW, rng=self._rng)
            alive = hp > 0
            nxt = tables.boss[g][min(window + 1, BOSS_WINDOWS) - 1][li]                 # (N, Nb)
            we = interp_matrix((hp + shield) / build.hp, grid)
            wb = interp_matrix(np.maximum(ehp, 0.0) / boss_max, tables.boss_grid)
            win_value = tables.config.boss_value + float(np.interp(1.0, grid, tables.ready[g][li]))
            after = np.where(ehp <= 0, win_value, np.einsum("sn,nb,sb->s", we, nxt, wb))
        else:
            rules = self.mcfg.mdp.rules
            first, rounds = fight_segment(enemy, rules)
            hp, shield, ehp, *_ = fight(build, hp, shield, enemy, boss=False, ehp=ehp, start_round=first,
                                        rounds=rounds, rng=self._rng)
            alive = hp > 0
            e1 = (hp + shield) / build.hp
            after = np.interp(e1, grid, tables.ready[g][li])
            if pauses_after_segment(enemy, rules):
                pause = self._elite_pause(state, tables, li, build, enemy, cost)        # (N, Nb)
                elite_max = float(enemy.get("max_hp") or enemy["hp"])
                we = interp_matrix(e1, grid)
                wb = interp_matrix(np.maximum(ehp, 0.0) / elite_max, tables.boss_grid)
                after = np.where(ehp > 0, np.einsum("sn,nb,sb->s", we, pause, wb), after)
        value = self._per_point(start, np.where(alive, after, -cost), len(grid))
        value[0] = -cost
        value = isotonic(value)
        death = 1.0 - self._per_point(start, alive.astype(float), len(grid))
        death[0] = 1.0
        return value, death

    def _boss_outlook(self, state: dict, floor: int) -> float:
        """Expected boss kills ahead of `floor`, chaining the next bosses' win chances (P1 + P1 P2 + ...).
        Each chance assumes a full heal before every window and the lasting build: an upper bound, so the
        extraction it drives only fires when even perfect healing would not get past the coming bosses."""
        build = quantize(self._calibrated(self.lasting_build(state)))
        map_id = state["run"].get("map") or "mall"
        total, chain = 0.0, 1.0
        for boss_floor in next_boss_floors(floor, self.mcfg.outlook_bosses):
            key = (build, map_id, boss_floor)
            if key not in self._boss_odds:
                self._boss_odds[key] = boss_win_chance(build, foe_at("boss", boss_floor, map_id, build))
            chain *= self._boss_odds[key]
            total += chain
        return total

    def _hopeless_boss(self, state: dict, ctr: dict, build: Build, enemy: dict, summary: str) -> Decision | None:
        """Bank the bag instead of fighting a boss the build cannot realistically beat: extract (offered during
        a boss fight), else a smoke bomb, which ends the run as an extraction too."""
        run = state["run"]
        if self.mcfg.extract_below <= 0 or self.greedy(run):
            return None
        win = boss_win_chance(build, {**enemy, "hp": float(enemy.get("max_hp") or enemy["hp"])},
                              ehp=float(enemy["hp"]), start_round=int(enemy.get("rounds_fought") or 0) + 1)
        outlook = win * (1.0 + self._boss_outlook(state, run["floor"]))
        if outlook >= self.mcfg.extract_below:
            return None
        why = f"{summary}: win {win:.1%} even healing every window, expected bosses {outlook:.2f}"
        if "extract" in ctr:
            return Decision("extract", {}, f"{why} -> extract and bank the bag")
        smoke = self._bag_item(run, "smoke")
        if smoke is not None and self._allowed(run["floor"], "use", {"index": smoke["index"]}):
            return Decision("use", {"index": smoke["index"]}, f"{why} -> smoke out (ends the run as an extraction)")
        return None

    def _elite_pause(self, state: dict, tables: ValueTables, li: int, build: Build, enemy: dict,
                     death_cost: float = 0.0) -> np.ndarray:
        """Values at the coming elite pause under price index li, [hp point, elite-hp point]: heals from
        the live inventory, then the remaining rounds against this elite. Guard / burst charges are
        assumed spent by then (conservative)."""
        run, grid = state["run"], tables.grid
        heal = _Closure([o.item for o in self._heal_options(run, True)], grid, float(run["max_hp"]),
                        np.array([float(tables.lambdas[li])]), self.mcfg.mdp.heal_iterations)
        foe = {**enemy, "hp": float(enemy.get("max_hp") or enemy["hp"]), "rounds_fought": 0}
        later = replace(build, guard_hits=0, burst_hits=0)
        lands = elite_pause_kernel(later, foe, grid, tables.boss_grid, self.mcfg.pause_samples, self._rng,
                                   self.mcfg.mdp.rules)
        ready = tables.ready[run["floor"]][li][None, :]
        pause = elite_pause_value(lands, ready, heal, len(tables.boss_grid), death_cost)[0]
        if tables.config.gold:                 # gold objective: extracting at the pause banks the bag
            pause = np.maximum(pause, 0.0)
            pause[0, :] = -death_cost
        return pause

    def _encounter(self, state: dict, ctr: dict, tables: ValueTables) -> Decision:
        """Heal / field buff / grenade / smoke / flee / fight, whichever gives the most expected bosses.
        Heals and smoke bombs are priced by their run budgets; field buffs are free but must gain
        `prep_margin`. Every option's value already includes the best heals that could follow it."""
        run = state["run"]
        enemy = battle_enemy(state, run["event"]["enemy"])
        boss = bool(enemy.get("boss"))
        g, e, max_hp, grid = run["floor"], effective_hp(run), float(run["max_hp"]), tables.grid
        kind = "boss" if boss else "elite" if enemy.get("elite") else "normal"
        stock = heal_stock(run)
        if boss:
            window = int(enemy.get("rounds_fought") or 0) // BOSS_WINDOW + 1
            values = tables.boss_at(g, window, e, enemy["hp"] / max(1.0, float(enemy.get("max_hp") or enemy["hp"])))
        else:
            values = tables.at(tables.encounter[kind], g, e)
        smoke = None if boss else self._bag_item(run, "smoke")   # on a boss a smoke ends the run
        if smoke is not None and not self._allowed(g, "use", {"index": smoke["index"]}):
            smoke = None
        smokes = smoke_count(run)
        li = tables.price_index(values, stock, smokes)
        lam, mu = float(tables.lambdas[li]), float(tables.smoke_prices[li])
        heals = self._heal_options(run, True)
        closure = _Closure([o.item for o in heals], grid, max_hp, np.array([lam]), self.mcfg.mdp.heal_iterations)

        def with_heals(curve: np.ndarray) -> float:
            return float(np.interp(e, grid, closure(curve[None, :])[0]))

        build = self._live_build(state)
        fight_curve, death_curve = self._fight_curve(state, tables, li, build, enemy)
        death = float(np.interp(e, grid, death_curve))
        tag = f"{kind} {enemy.get('name')}"
        summary = f"{tag}: death {death:.1%}, lam {lam:g}, stock {stock:.1f}, smoke {smokes} @ {mu:g}"
        if boss:
            bank = self._hopeless_boss(state, ctr, build, enemy, summary)
            if bank:
                return bank

        # options that do not consume a priced heal: fight now, flee, field buffs, grenade
        curve = fight_curve
        flee_q = -np.inf
        if "flee" in ctr and not boss:
            m = re.search(r"(\d+)%", ctr["flee"].get("label", ""))
            p = int(m.group(1)) / 100 if m else flee_chance(build, g)
            fail = interp_matrix(np.maximum(grid - flee_hit(build, enemy), 0.0), grid) @ fight_curve
            flee_curve = p * tables.ready[g][li] + (1 - p) * fail
            flee_curve[0] = 0.0
            flee_q = float(np.interp(e, grid, flee_curve))
            curve = np.maximum(fight_curve, flee_curve)
        q_base = with_heals(curve)
        best_prep, q_prep = None, -np.inf
        if boss or death >= self.mcfg.prep_min_death:   # a safe regular fight cannot gain from a buff
            best_prep, q_prep = self._best_preparation(state, tables, li, build, enemy, with_heals)

        option, q_heal, q_go = self._choose(e, lam, max_hp, curve, grid, heals)
        q_smoke = float(np.interp(e, grid, tables.ready[g][li])) - mu if smoke is not None else -np.inf
        healed = max([e, *(float(o.item.apply(np.array([e]), max_hp)[0]) for o in heals)])
        if self._wish_ready(state) and float(np.interp(healed, grid, death_curve)) >= self.mcfg.wish_min_death:
            ready = tables.ready[g][li]
            q_wish = (tables.config.boss_value + float(np.interp(1.0, grid, ready)) if boss
                      else float(np.interp(e, grid, ready)))
            q_other = max(q_base, q_heal, q_prep, q_smoke)
            if q_wish - q_other >= self.mcfg.wish_gain * q_wish:
                self._wish_sent = True
                return Decision("field_use", {"id": WISH},
                                f"{summary} -> wish kill ({q_wish:.2f} vs {q_other:.2f})")
        q_other = max(q_base, q_heal, q_prep, q_smoke)
        if tables.config.gold and "extract" in ctr and q_other < self.mcfg.heal_margin:   # nothing left to gain
            return Decision("extract", {}, f"{summary} -> extract: banking {bag_at_risk(run):.1f}M is worth "
                                           f"more than any way on ({q_other:.2f})")
        if smoke is not None and q_smoke > max(q_base, q_heal, q_prep) + self.mcfg.heal_margin:
            return Decision("use", {"index": smoke["index"]},
                            f"{summary} -> smoke ({q_smoke:.2f} vs {max(q_base, q_heal, q_prep):.2f})")
        if best_prep is not None and q_prep > max(q_base, q_heal) + self.mcfg.prep_margin:
            return Decision(best_prep[0], best_prep[1], f"{summary} -> {best_prep[2]} ({q_prep:.2f} vs {q_base:.2f})")
        if option and q_heal > q_go + self.mcfg.heal_margin:
            return Decision(option.decision.command, option.decision.params,
                            f"{summary} -> {option.decision.reason} ({q_heal:.2f} vs {q_go:.2f})")
        fight_q = float(np.interp(e, grid, fight_curve))
        if flee_q > fight_q + self.mcfg.heal_margin:
            return Decision("flee", {}, f"{summary} -> flee ({flee_q:.2f} vs {fight_q:.2f})")
        return Decision("fight", {}, f"{summary} -> fight ({fight_q:.2f})")

    def _wish_ready(self, state: dict) -> bool:
        """A wish in the field stash, and nothing it would wipe that matters: no still-missing collectible,
        no looted relic bag being banked, and less than wish_max_loot of loot in the bag and safe box."""
        run = state["run"]
        if not any(it.get("id") == WISH and (it.get("quantity") or 0) > 0 for it in run.get("field") or []):
            return False
        if not self._allowed(run["floor"], "field_use", {"id": WISH}):
            return False
        held = [*(run.get("bag") or []), *(run.get("safe") or [])]
        if any(it.get("id") in self.missing_collectibles(state) for it in held):
            return False
        loot = sum(float(it.get("value") or 0) * (it.get("quantity") or 1) for it in held if it.get("id") not in KEEP_IN_BAG)
        return loot < self.mcfg.wish_max_loot and not self.banking

    def _decide(self, state: dict) -> Decision:
        sent, self._wish_sent = self._wish_sent, False
        preview = state.get("wish_preview") or {}
        if sent and preview.get("token") and (state["run"].get("event") or {}).get("enemy"):
            return Decision("wish_confirm", {"token": preview["token"]},
                            f"confirm wish on {preview.get('enemy')} (wipes {preview.get('value', 0):,} of run loot)")
        if sent:   # no preview came back: do not ask again on this floor
            self.banned.add((state["run"]["floor"], "field_use", repr(sorted({"id": WISH}.items()))))
        return super()._decide(state)

    def _best_preparation(self, state: dict, tables: ValueTables, li: int, build: Build, enemy: dict,
                          with_heals: Callable[[np.ndarray], float]) -> tuple[tuple | None, float]:
        """Best first step among field buffs, pairs of them and a grenade, valued with heals after it.
        Returns ((command, params, label) or None, value)."""
        run = state["run"]
        floor = run["floor"]
        preps = [p for p in field_items.fight_preparations(run)
                 if self._allowed(floor, "field_use", {"id": p.id})]
        scored: list[tuple[float, FieldPrep, Build, dict]] = []
        for prep in preps:
            b, mods = field_items.apply(run, build, prep)
            scored.append((with_heals(self._fight_curve(state, tables, li, b, enemy)[0]), prep, b, mods))
        scored.sort(key=lambda x: -x[0])
        best, best_q = None, -np.inf
        if scored:
            q, prep = scored[0][0], scored[0][1]
            best, best_q = ("field_use", {"id": prep.id}, f"field {prep.name}"), q
        top = scored[: self.mcfg.prep_pairs]
        for i, (_, first, b1, mods1) in enumerate(top):
            for _, second, _, _ in top[i + 1:]:
                if not field_items.compatible(first, second, run):
                    continue
                b2, _ = field_items.apply(run, b1, second, mods1)
                q = with_heals(self._fight_curve(state, tables, li, b2, enemy)[0])
                if q > best_q:
                    best, best_q = ("field_use", {"id": first.id}, f"field {first.name} (+ {second.name})"), q
        grenade = self._bag_item(run, "grenade")
        boss_capped = enemy.get("boss") and (enemy.get("grenades", 0) >= 2 or enemy["hp"] <= 1)
        if grenade and not boss_capped and self._allowed(floor, "use", {"index": grenade["index"]}):
            cut = max(1, int(enemy["hp"] * (0.12 if enemy.get("boss") else 0.30)))
            weakened = {**enemy, "hp": max(1 if enemy.get("boss") else 0, enemy["hp"] - cut)}
            q = with_heals(self._fight_curve(state, tables, li, build, weakened)[0])
            if q > best_q:
                best, best_q = ("use", {"index": grenade["index"]}, "grenade"), q
        return best, best_q

    # ------------------------------------------------------------------ between floors
    def _advance(self, state: dict, ctr: dict) -> Decision:
        run = state["run"]
        tables = self._tables(state)
        if tables is None:
            return super()._advance(state, ctr)
        if self.cfg.max_floor and run["floor"] >= self.cfg.max_floor and "extract" in ctr:
            return Decision("extract", {}, f"reached max floor {self.cfg.max_floor}")
        g, e, max_hp = run["floor"], effective_hp(run), float(run["max_hp"])
        stock = heal_stock(run)
        li = tables.price_index(tables.at(tables.ready, g, e), stock, smoke_count(run))
        lam, grid = float(tables.lambdas[li]), tables.grid
        q_step = float(np.interp(e, grid, tables.step[g][li]))
        q_rush = float(np.interp(e, grid, tables.rush[g][li]))
        elites = self.mcfg.mdp.rush_elites and "rush_auto_elite" in (state.get("player") or {})
        q_elite = float(np.interp(e, grid, tables.rush_elite[g][li])) if elites else -np.inf
        summary = (f"step {q_step:.2f} rush {q_rush:.2f}" + (f" (with elites {q_elite:.2f})" if elites else "")
                   + f", lam {lam:g}, stock {stock:.1f}")
        if self.mcfg.extract_below > 0 and "extract" in ctr and not self.greedy(run):
            outlook = self._boss_outlook(state, g)
            if outlook < self.mcfg.extract_below:
                return Decision("extract", {}, f"{summary}: expected further bosses {outlook:.2f} < "
                                              f"{self.mcfg.extract_below:g} -> extract and bank the bag")
        curve = np.maximum(tables.step[g][li], tables.rush[g][li])
        if elites:
            curve = np.maximum(curve, tables.rush_elite[g][li])
        option, q_heal, q_go = self._choose(e, lam, max_hp, curve, grid, self._heal_options(run, False))
        if tables.config.gold and "extract" in ctr and max(q_go, q_heal) < self.mcfg.heal_margin:
            return Decision("extract", {}, f"{summary}: banking {bag_at_risk(run):.1f}M beats going on "
                                           f"({max(q_go, q_heal):.2f}) -> extract")
        if option and q_heal > q_go + self.mcfg.heal_margin:
            return Decision(option.decision.command, option.decision.params,
                            f"{summary} -> {option.decision.reason} ({q_heal:.2f})")
        if "rush" in ctr and "down" in ctr and entry_heal_pending(run):
            return Decision("down", {}, summary + " -> step (healing on floor entry pending: a rush would"
                                                  " auto-fight once it lifts HP above 50%)")
        if "rush" in ctr:
            auto = q_elite >= max(q_step, q_rush) - self.mcfg.elite_rush_tolerance
            if auto or q_rush >= q_step - self.mcfg.rush_tolerance or "down" not in ctr:
                return self._rush(state, ctr, auto, summary + " -> rush")
        if "down" in ctr:
            return Decision("down", {}, summary + " -> step")
        return super()._advance(state, ctr)
