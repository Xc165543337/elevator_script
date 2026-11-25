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
from dataclasses import dataclass, field, fields, replace
from typing import Callable

import numpy as np

import field as field_items
from combat import Calibration, battle_enemy
from field import FieldPrep
from mdp import (BOSS_WINDOWS, FIELD_HEALS, SHOP_HEALS, HealItem, MdpConfig, ValueTables, _Closure,
                 elite_pause_kernel, elite_pause_value, flee_chance, flee_hit, interp_matrix, load_tables,
                 save_tables, solve, tables_key, with_calibration)
from model import BOSS_WINDOW, Build, fight, fight_segment, pauses_after_segment
from policy import KEEP_IN_BAG, BossFarmPolicy, Decision, PolicyConfig

WISH = "field_wish"   # 为了我所期待的


@dataclass
class MarkovConfig:
    from_floor: int = 30            # shallower floors are trivial and the build is still ramping up
    # expected bosses a rush may give up against single steps; it is paid on every rush of the run, so it
    # only absorbs Monte Carlo noise (0.02 let a two-fight rush at 3.6% death risk through at F1205 live)
    rush_tolerance: float = 0.002
    heal_margin: float = 0.002      # a heal / flee must beat the alternative by this many bosses
    prep_margin: float = 0.01       # a field buff / grenade (no budget price) must gain this many bosses
    prep_pairs: int = 3             # best single preparations combined pairwise (buff + buff synergies)
    prep_min_death: float = 0.001   # search field preparations only against regular fights at least this deadly
    extract_below: float = 0.0      # extract once expected further bosses drop below this (0 = never)
    live_samples: int = 300         # Monte Carlo fights per HP grid point for live evaluations
    pause_samples: int = 48         # live fights per (hp, elite hp) point when valuing the elite pause
    # 为了我所期待的 (one per run at F500, 0.1% per floor): kills the current enemy, wipes this run's loot.
    # Only for a real emergency: the coming fight segment must stay at least wish_min_death deadly even
    # after the best heal at hand (a boss that cannot kill from full HP in one window is healed through,
    # not wished away), and the wish must add wish_gain of the value at stake over the best alternative.
    wish_gain: float = 0.15
    wish_min_death: float = 0.05
    wish_max_loot: int | None = None   # never wipe more run loot than this; None = PolicyConfig.relic_loot_value
    build_tolerance: float = 0.10   # relative stat drift before re-solving (the live fight always uses the exact build)
    cache_dir: str = "cache"
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
    """Value tables for the current build: in memory, then the disk cache, then a fresh solve."""

    def __init__(self, cfg: MarkovConfig, log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.log = log or (lambda _msg: None)
        self._tables: ValueTables | None = None

    def tables(self, build: Build, map_id: str) -> ValueTables:
        current = self._tables
        if current is not None and current.map_id == map_id and self._close(current.build, build):
            return current
        build = quantize(build)
        key = tables_key(build, map_id, self.cfg.mdp)
        path = os.path.join(self.cfg.cache_dir, f"mdp_{key}.npz")
        if os.path.exists(path):
            self._tables = load_tables(path, build, map_id, self.cfg.mdp)
        else:
            self.log(f"mdp: solving tables for {map_id} (atk {build.atk:.0f}, def {build.defense:.0f}, "
                     f"agi {build.agi:.0f}, hp {build.hp:.0f})")
            self._tables = solve(build, map_id, self.cfg.mdp, progress=self.log)
            os.makedirs(self.cfg.cache_dir, exist_ok=True)
            save_tables(self._tables, path)
            self.log(f"mdp: solved in {self._tables.solve_seconds:.0f}s -> {path}")
        return self._tables

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
        # one source of truth for the server's round limits: the policy config
        self.mcfg = replace(markov, mdp=replace(markov.mdp, rules=self.cfg.rules))
        self.planner = Planner(self.mcfg, log)
        self._rng = np.random.default_rng()
        self._wish_sent = False   # a wish preview was requested; confirm it on the next decision

    # ------------------------------------------------------------------ helpers
    def _calibrated(self, build: Build) -> Build:
        kp, ke = self.cal.conservative()
        return with_calibration(build, kp, ke)

    def _tables(self, state: dict) -> ValueTables | None:
        run = state["run"]
        if run["floor"] < self.mcfg.from_floor:
            return None
        tables = self.planner.tables(self._calibrated(self._table_build(state)), run.get("map") or "mall")
        return tables if tables.in_range(run["floor"]) else None

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
            win_value = 1.0 + float(np.interp(1.0, grid, tables.ready[g][li]))
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
                pause = self._elite_pause(state, tables, li, build, enemy)              # (N, Nb)
                elite_max = float(enemy.get("max_hp") or enemy["hp"])
                we = interp_matrix(e1, grid)
                wb = interp_matrix(np.maximum(ehp, 0.0) / elite_max, tables.boss_grid)
                after = np.where(ehp > 0, np.einsum("sn,nb,sb->s", we, pause, wb), after)
        value = isotonic(self._per_point(start, np.where(alive, after, 0.0), len(grid)))
        death = 1.0 - self._per_point(start, alive.astype(float), len(grid))
        death[0] = 1.0
        return value, death

    def _elite_pause(self, state: dict, tables: ValueTables, li: int, build: Build, enemy: dict) -> np.ndarray:
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
        return elite_pause_value(lands, ready, heal, len(tables.boss_grid))[0]

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
            q_wish = 1.0 + float(np.interp(1.0, grid, ready)) if boss else float(np.interp(e, grid, ready))
            q_other = max(q_base, q_heal, q_prep, q_smoke)
            if q_wish - q_other >= self.mcfg.wish_gain * q_wish:
                self._wish_sent = True
                return Decision("field_use", {"id": WISH},
                                f"{summary} -> wish kill ({q_wish:.2f} vs {q_other:.2f})")
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
        no looted relic bag being banked, and less loot in the bag and safe box than the wish loot limit."""
        run = state["run"]
        if not any(it.get("id") == WISH and (it.get("quantity") or 0) > 0 for it in run.get("field") or []):
            return False
        if not self._allowed(run["floor"], "field_use", {"id": WISH}):
            return False
        held = [*(run.get("bag") or []), *(run.get("safe") or [])]
        if any(it.get("id") in self.missing_collectibles(state) for it in held):
            return False
        loot = sum(float(it.get("value") or 0) * (it.get("quantity") or 1) for it in held if it.get("id") not in KEEP_IN_BAG)
        limit = self.cfg.relic_loot_value if self.mcfg.wish_max_loot is None else self.mcfg.wish_max_loot
        return loot < limit and not self.banking

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
        summary = f"step {q_step:.2f} rush {q_rush:.2f}, lam {lam:g}, stock {stock:.1f}"
        if (self.mcfg.extract_below > 0 and "extract" in ctr and not self.greedy(run)
                and max(q_step, q_rush) < self.mcfg.extract_below):
            return Decision("extract", {}, f"{summary}: below {self.mcfg.extract_below} expected bosses -> extract")
        curve = np.maximum(tables.step[g][li], tables.rush[g][li])
        option, q_heal, q_go = self._choose(e, lam, max_hp, curve, grid, self._heal_options(run, False))
        if option and q_heal > q_go + self.mcfg.heal_margin:
            return Decision(option.decision.command, option.decision.params,
                            f"{summary} -> {option.decision.reason} ({q_heal:.2f})")
        if "rush" in ctr and "down" in ctr and entry_heal_pending(run):
            return Decision("down", {}, summary + " -> step (healing on floor entry pending: a rush would"
                                                  " auto-fight once it lifts HP above 50%)")
        if "rush" in ctr and (q_rush >= q_step - self.mcfg.rush_tolerance or "down" not in ctr):
            return Decision("rush", {}, summary + " -> rush")
        if "down" in ctr:
            return Decision("down", {}, summary + " -> step")
        return super()._advance(state, ctr)
