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

from combat import Calibration
from mdp import (BOSS_WINDOWS, FIELD_HEALS, SHOP_HEALS, HealItem, MdpConfig, ValueTables, _Closure,
                 flee_chance, flee_hit, interp_matrix, load_tables, save_tables, solve, tables_key,
                 with_calibration)
from model import BOSS_WINDOW, REGULAR_ROUND_CAP, Build, fight
from policy import BossFarmPolicy, Decision, PolicyConfig


@dataclass
class MarkovConfig:
    from_floor: int = 30            # shallower floors are trivial and the build is still ramping up
    rush_tolerance: float = 0.02    # expected bosses a rush may give up against single steps
    heal_margin: float = 0.002      # a heal / flee must beat the alternative by this many bosses
    extract_below: float = 0.0      # extract once expected further bosses drop below this (0 = never)
    live_samples: int = 300         # Monte Carlo fights per HP grid point for live evaluations
    build_tolerance: float = 0.03   # relative stat drift before the tables are re-solved
    cache_dir: str = "cache"
    mdp: MdpConfig = field(default_factory=MdpConfig)


@dataclass(frozen=True)
class HealOption:
    item: HealItem
    decision: Decision


_HEAL_TEXT = re.compile(r"回复 (\d+)% 最大生命")


def _field_heal_spec(item: dict) -> HealItem | None:
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
        self.mcfg = markov or MarkovConfig()
        self.planner = Planner(self.mcfg, log)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ helpers
    def _calibrated(self, build: Build) -> Build:
        kp, ke = self.cal.conservative()
        return with_calibration(build, kp, ke)

    def _tables(self, state: dict) -> ValueTables | None:
        run = state["run"]
        if run["floor"] < self.mcfg.from_floor:
            return None
        tables = self.planner.tables(self._calibrated(self.lasting_build(state)), run.get("map") or "mall")
        return tables if tables.in_range(run["floor"]) else None

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

    def _free_help(self, run: dict, boss: bool, death: float, tag: str) -> Decision | None:
        """Field buffs and grenades are free: use them when the coming fight is still risky."""
        limit = self.cfg.max_boss_window_risk if boss else self.cfg.max_death_risk
        if death <= limit:
            return None
        buff = self._combat_buff(run, boss)
        if buff:
            buff.reason = f"{tag}: death {death:.1%} -> {buff.reason}"
            return buff
        grenade = self._bag_item(run, "grenade")
        if grenade and self._allowed(run["floor"], "use", {"index": grenade["index"]}):
            return Decision("use", {"index": grenade["index"]}, f"{tag}: grenade (death {death:.1%})")
        return None

    # ------------------------------------------------------------------ encounters
    def _fight(self, state: dict, ctr: dict) -> Decision:
        run = state["run"]
        enemy = run["event"]["enemy"]
        tables = None if enemy.get("ambush") else self._tables(state)
        if tables is None:
            return super()._fight(state, ctr)
        if enemy.get("boss"):
            return self._boss(state, tables)
        return self._regular(state, ctr, tables)

    def _regular(self, state: dict, ctr: dict, tables: ValueTables) -> Decision:
        run = state["run"]
        enemy = run["event"]["enemy"]
        g, e, max_hp = run["floor"], effective_hp(run), float(run["max_hp"])
        kind = "elite" if enemy.get("elite") else "normal"
        tag = f"{kind} {enemy.get('name')}"
        stock = heal_stock(run)
        li = tables.price_index(tables.at(tables.encounter[kind], g, e), stock)
        lam, grid = float(tables.lambdas[li]), tables.grid
        ready = tables.ready[g][li]
        build = self._calibrated(self.current_build(state))

        start, hp, shield = self._sim_starts(grid, build)
        hp, shield, *_ = fight(build, hp, shield, enemy, boss=False, ehp=np.full(len(start), float(enemy["hp"])),
                               rounds=REGULAR_ROUND_CAP, rng=self._rng)
        alive = hp > 0
        after = np.where(alive, np.interp((hp + shield) / build.hp, grid, ready), 0.0)
        fight_curve = isotonic(self._per_point(start, after, len(grid)))
        death_curve = 1.0 - self._per_point(start, alive.astype(float), len(grid))
        death_curve[0] = 1.0
        curve = fight_curve
        flee_q = -np.inf
        if "flee" in ctr:
            m = re.search(r"(\d+)%", ctr["flee"].get("label", ""))
            p = int(m.group(1)) / 100 if m else flee_chance(build, g)
            fail = interp_matrix(np.maximum(grid - flee_hit(build, enemy), 0.0), grid) @ fight_curve
            flee_curve = p * ready + (1 - p) * fail
            flee_curve[0] = 0.0
            flee_q = float(np.interp(e, grid, flee_curve))
            curve = np.maximum(fight_curve, flee_curve)
        option, q_heal, q_go = self._choose(e, lam, max_hp, curve, grid, self._heal_options(run, True))
        fight_q = float(np.interp(e, grid, fight_curve))
        death = float(np.interp(e, grid, death_curve))
        summary = f"{tag}: fight {fight_q:.2f} (death {death:.1%}), lam {lam:g}, stock {stock:.1f}"
        if option and q_heal > q_go + self.mcfg.heal_margin:
            return Decision(option.decision.command, option.decision.params,
                            f"{summary} -> {option.decision.reason} ({q_heal:.2f})")
        if flee_q > fight_q + self.mcfg.heal_margin:
            return Decision("flee", {}, f"{summary} -> flee ({flee_q:.2f})")
        help_ = self._free_help(run, False, death, tag)
        if help_:
            return help_
        return Decision("fight", {}, summary)

    def _boss(self, state: dict, tables: ValueTables) -> Decision:
        run = state["run"]
        enemy = run["event"]["enemy"]
        g, e, max_hp = run["floor"], effective_hp(run), float(run["max_hp"])
        fought = int(enemy.get("rounds_fought") or 0)
        window = fought // BOSS_WINDOW + 1
        boss_frac = enemy["hp"] / max(1.0, float(enemy.get("max_hp") or enemy["hp"]))
        tag = f"boss {enemy.get('name')} w{window}"
        stock = heal_stock(run)
        li = tables.price_index(tables.boss_at(g, window, e, boss_frac), stock)
        lam, grid = float(tables.lambdas[li]), tables.grid
        next_values = tables.boss[g][min(window + 1, BOSS_WINDOWS) - 1][li]          # (N, Nb)
        win_value = 1.0 + float(np.interp(1.0, grid, tables.ready[g][li]))
        build = self._calibrated(self.current_build(state))

        start, hp, shield = self._sim_starts(grid, build)
        rounds = BOSS_WINDOW - fought % BOSS_WINDOW
        hp, shield, ehp, *_ = fight(build, hp, shield, enemy, boss=True, ehp=np.full(len(start), float(enemy["hp"])),
                                    start_round=fought + 1, rounds=rounds, rng=self._rng)
        alive = hp > 0
        won = alive & (ehp <= 0)
        going = alive & (ehp > 0)
        we = interp_matrix((hp + shield) / build.hp, grid)
        wb = interp_matrix(np.maximum(ehp, 0.0) / float(enemy.get("max_hp") or enemy["hp"]), tables.boss_grid)
        cont = np.einsum("sn,nb,sb->s", we, next_values, wb)
        after = np.where(won, win_value, np.where(going, cont, 0.0))
        curve = isotonic(self._per_point(start, after, len(grid)))
        death_curve = 1.0 - self._per_point(start, alive.astype(float), len(grid))
        death_curve[0] = 1.0
        option, q_heal, q_go = self._choose(e, lam, max_hp, curve, grid, self._heal_options(run, True))
        death = float(np.interp(e, grid, death_curve))
        summary = f"{tag}: window {q_go:.2f} (death {death:.1%}), lam {lam:g}, stock {stock:.1f}"
        if option and q_heal > q_go + self.mcfg.heal_margin:
            return Decision(option.decision.command, option.decision.params,
                            f"{summary} -> {option.decision.reason} ({q_heal:.2f})")
        help_ = self._free_help(run, True, death, tag)
        if help_:
            return help_
        return Decision("fight", {}, summary)

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
        li = tables.price_index(tables.at(tables.ready, g, e), stock)
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
        if "rush" in ctr and (q_rush >= q_step - self.mcfg.rush_tolerance or "down" not in ctr):
            return Decision("rush", {}, summary + " -> rush")
        if "down" in ctr:
            return Decision("down", {}, summary + " -> step")
        return super()._advance(state, ctr)
