"""Markov decision process for boss farming, solved by backward induction.

Objective: maximise the expected number of formal boss kills of one run (each kill is an independent
0.5% rainbow roll). Death and extraction both end the run, so neither carries value of its own.

State (per floor g): effective HP `e` = (HP + shield) / max HP on a grid over [0, E_MAX], plus the
decision point:
  * ready      - nothing pending: heal (non-combat items), step one floor, or rush (with the server's
                 自动打精英 setting off: stop at every elite; or on: fight its first 8 rounds in the rush);
  * encounter  - a normal monster or elite is waiting: heal (any item), fight, or flee;
  * elite pause - (live rules) after round 8 of an elite fight, with elite HP fraction b: heal, then
                 fight the remaining rounds (solved on the fly, not stored);
  * boss       - before boss window w (1..4, window 4+ is stationary) with boss HP fraction b.

Healing stock is a run-long budget shared by every decision. Tracking it exactly would multiply the
state space by the inventory, so the budget constraint is relaxed with a Lagrange multiplier: for a
price `lam` (bosses per max-HP unit of healing) each heal costs `lam * units`, and found field heals
earn `lam * units`. Tables are solved for a grid of prices at once. At decision time the price is
chosen by the dual bound  V(s, H) ~= min_lam V_lam(s) + lam * H  for the live heal stock H, and the
action is the best one under that price. This is re-evaluated at every decision, so the price rises
as the stock runs down. Smoke bombs (leave a normal / elite encounter unharmed) are a second budget with
their own price mu; tables are solved for every (lambda, mu) pair of the two grids.

Transitions come from the same combat model as everything else (model.fight, Monte Carlo per grid
point); encounter odds from model.floor_mix. Events, loot and relics leave HP unchanged here (the live
policy still handles their content).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import astuple, dataclass, field, replace
from typing import Callable

import numpy as np

from model import (BOSS_WINDOW, LIVE_RULES, REGULAR_ROUND_CAP, Build, ServerRules, enemy, fight, floor_mix,
                   foe_at, is_boss_floor)

E_MAX = 1.25                 # effective HP cap: full HP plus the 25% shield of a barrier serum
BOSS_WINDOWS = 4             # windows 1-3 differ (first-round bonus, berserk ramp); 4+ is stationary
RUSH_FLOORS = 10             # floors a rush descends at most (server setting)
RUSH_STOP_HP = 0.5           # the server stops a rush at HP <= 50%
FIELD_DROP_CHANCE = 0.52     # server FIELD_CHANCE; guaranteed on F1, boss floors and the floor before

# Gold objective: expected run loot (bag + safe box, million gold) by floor, measured in the game engine
# with the threshold agent (3 runs per map, real mode). A bag holds about a thousand items worth 2k-48k
# gold, so its value at a floor varies little between runs and a curve stands in for it. Beyond the
# last point the curve goes on with the slope of its last segment.
LOOT_CURVES = {
    "mall": ((0, 0.0), (50, 0.49), (100, 0.75), (200, 1.71), (300, 3.44), (500, 6.11), (800, 10.14),
             (1000, 12.42), (1200, 14.08), (1500, 15.99)),
    "subway": ((0, 0.0), (50, 0.66), (100, 1.41), (200, 2.93), (300, 4.39), (500, 7.40), (800, 10.12)),
    "abyss": ((0, 0.0), (50, 1.02), (100, 2.12), (200, 4.27), (300, 6.22), (500, 8.47)),
    "auction": ((0, 0.0), (25, 0.68), (50, 1.23), (100, 2.39), (150, 3.78), (200, 4.77), (300, 6.22), (400, 7.32),
                (500, 8.14), (600, 8.78)),
    "tide": ((0, 0.0), (25, 0.58), (50, 1.16), (100, 2.14), (150, 3.10), (200, 3.82), (300, 4.94), (400, 5.71)),
    "zenith": ((0, 0.0), (25, 0.60), (50, 1.18), (100, 2.20), (150, 3.11), (200, 4.14)),
}
SAFE_KEPT = 0.2              # million gold the 4-slot safe box keeps on death (best four items, measured)


def loot_at(map_id: str, floor: int) -> float:
    """Expected run loot at `floor` in million gold (LOOT_CURVES; unmeasured maps use the mall curve)."""
    points = LOOT_CURVES.get(map_id, LOOT_CURVES["mall"])
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points])
    if floor <= xs[-1]:
        return float(np.interp(floor, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + max(0.0, slope) * (floor - xs[-1]))


def loot_profile(config: "MdpConfig", map_id: str, top: int) -> tuple[np.ndarray, np.ndarray]:
    """(gain, risk) per floor 0..top+2 in model units: loot found on arriving at a floor, and what a death
    on that floor loses (the bag outside the safe box). Zeros unless config.gold."""
    if not config.gold:
        return np.zeros(top + 3), np.zeros(top + 3)
    loot = np.array([loot_at(map_id, g) for g in range(top + 3)])
    return np.diff(loot, prepend=0.0), np.maximum(loot - SAFE_KEPT, 0.0)


def death_of(*kernels: np.ndarray) -> np.ndarray:
    """P(death) per start state of a kernel split into survival parts (rows of kill / draw / cont / lands)."""
    return np.clip(1.0 - sum(k.sum(axis=1) for k in kernels), 0.0, 1.0)
MODEL_VERSION = 8            # bump when the dynamics change, to invalidate cached tables


# --------------------------------------------------------------------------- heal items
@dataclass(frozen=True)
class HealItem:
    """A consumable that raises effective HP. Values from the server item specs."""
    id: str
    heal: float = 0.0              # fraction of max HP
    flat: float = 0.0              # extra absolute HP
    shield_floor: float = 0.0      # shield raised to at least this fraction of max HP
    shield_add: float = 0.0        # absolute shield added
    combat_only: bool = False
    critical_at: float = 0.0       # if HP <= critical_at, heal critical_heal instead
    critical_heal: float = 0.0

    def units(self, max_hp: float) -> float:
        """Nominal size in max-HP units: what it costs from the heal budget."""
        return self.heal + (self.flat + self.shield_add) / max_hp + self.shield_floor

    def apply(self, e: np.ndarray, max_hp: float) -> np.ndarray:
        """Effective HP after use, for effective HP `e` (array)."""
        hp = np.minimum(e, 1.0)
        shield = e - hp
        heal = np.full_like(hp, self.heal)
        if self.critical_at:
            heal = np.where(hp <= self.critical_at, self.critical_heal, heal)
        hp = np.minimum(1.0, hp + heal + self.flat / max_hp)
        shield = np.maximum(shield, self.shield_floor) + self.shield_add / max_hp
        return np.minimum(E_MAX, hp + shield)


# Shop medicine (server content.json `medicine` + game.py coffee / ice tea). The slow serum's regen
# (+16% on each of the next 3 floors) is left out on purpose: the HP arrives too late for the fight
# that prompted the heal, so the model stays conservative.
SHOP_HEALS = {h.id: h for h in (
    HealItem("coffee", heal=0.40),
    HealItem("ice_tea", heal=0.25),
    HealItem("triage_vial", heal=0.55, flat=12),
    HealItem("barrier_serum", heal=0.50, shield_floor=0.25),
    HealItem("slow_serum", heal=0.38),
    HealItem("critical_serum", heal=0.55, critical_at=0.35, critical_heal=0.90),
    HealItem("field_surgery", heal=1.00, combat_only=True),
)}

# Instant field items (server `field_items`), keyed by in-game name.
FIELD_HEALS = {h.id: h for h in (
    HealItem("速凝绷带", heal=0.28),
    HealItem("试吃能量包", heal=0.18, shield_add=16),
    HealItem("折叠盾片", shield_add=38),
)}

# Mall field pool: name -> pick weight after the server's adjustments (heal / leech x1.5, a worn wear
# item x0.4 - the build wears two aprons). Only the healing entries need names; the rest is mass.
_MALL_FIELD_POOL = {"速凝绷带": 12, "试吃能量包": 12, "折叠盾片": 7, "_other": 60}


def field_heal_income(floor: int, max_hp: float) -> float:
    """Expected heal units found on entering `floor` (Mall pool)."""
    total = sum(_MALL_FIELD_POOL.values())
    per_drop = sum(w * FIELD_HEALS[name].units(max_hp) for name, w in _MALL_FIELD_POOL.items()
                   if name in FIELD_HEALS) / total
    guaranteed = floor == 1 or is_boss_floor(floor) or is_boss_floor(floor + 1)
    return per_drop * (1.0 if guaranteed else FIELD_DROP_CHANCE)


# --------------------------------------------------------------------------- grids
def interp_matrix(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Matrix W with (values_on_grid @ W.T)[k] == linear interpolation at x[k] (clamped to the grid)."""
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    out = np.zeros((len(x), len(grid)))
    rows = np.arange(len(x))
    out[rows, lo] += 1 - w
    out[rows, hi] += w
    return out


def _deposit(start: np.ndarray, x: np.ndarray, grid: np.ndarray, n_start: int, weight: float) -> np.ndarray:
    """Histogram of samples (start index, landing value x) split linearly onto `grid`."""
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    n = len(grid)
    flat = np.bincount(start * n + lo, weights=(1 - w) * weight, minlength=n_start * n)
    flat += np.bincount(start * n + hi, weights=w * weight, minlength=n_start * n)
    return flat.reshape(n_start, n)


# --------------------------------------------------------------------------- config & tables
@dataclass(frozen=True)
class MdpConfig:
    max_floor: int = 4000          # values beyond are 0; the full B06 set makes F2000+ reachable
    hp_points: int = 51            # effective-HP grid points over [0, E_MAX]
    boss_points: int = 11          # boss-HP grid points over (0, 1]
    samples: int = 256             # Monte Carlo fights per grid point
    kernel_stride: int = 4         # regular-fight kernels are shared by this many floors (strongest foe)
    boss_iterations: int = 60      # value-iteration sweeps of the stationary boss window (convergence
                                   # limit only: a boss never retreats, the fight goes on until one side dies)
    heal_iterations: int = 8       # heal chains considered per decision (closure depth)
    # heal prices in bosses per max-HP unit. Dense between 0.03 and 0.2: with (..., 0.05, 0.15, ...) the F2410 boss
    # (32 units left) priced a heal at 0.15 and the agent fought a 5.7% window instead of topping up; now 0.08.
    lambdas: tuple = (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 10.0)
    # smoke-bomb prices in bosses per bomb; the last one is prohibitive (= no smoke left). Dense where the
    # dual bound turns: a smoke saves roughly the heal price of the HP a deep fight costs (0.03-0.1); with
    # (0, 0.1, ...) a stock of 50 still priced at 0.1, and live runs ended with most smokes unused.
    smoke_prices: tuple = (0.0, 0.03, 0.05, 0.07, 0.1, 1000.0)
    pause_samples: int = 96        # Monte Carlo fights per (hp, elite hp) point of the elite-pause kernel
    rules: ServerRules = LIVE_RULES
    # also solve rushes with 自动打精英 on: the rush fights an elite up to its pause by itself while HP is
    # above 50%, continues on a kill and stops at the pause otherwise (game.py `rush`). Costs ~40% more time.
    rush_elites: bool = True
    # Objective. Bosses (default): +boss_value per formal boss kill. Gold: the unit is a million gold, a boss
    # kill is worth boss_value of it, loot found counts when found (loot_profile) and a death loses the bag
    # outside the safe box; extracting (between floors, in a fight, during a boss) banks it.
    gold: bool = False
    boss_value: float = 1.0
    seed: int = 0


@dataclass
class ValueTables:
    """Solved values, indexed [floor, price, hp point] (boss: [window, price, hp, boss hp]).
    A price index p stands for the pair (lambdas[p], smoke_prices[p]): heal price per max-HP unit and
    smoke-bomb price per bomb. Stored as float32."""
    config: MdpConfig
    build: Build
    map_id: str
    grid: np.ndarray
    boss_grid: np.ndarray
    lambdas: np.ndarray
    smoke_prices: np.ndarray
    ready: np.ndarray            # value at a ready state on floor g
    step: np.ndarray             # value of `down` from a ready state on floor g
    rush: np.ndarray             # value of `rush` from a ready state on floor g, stopping at elites
    rush_elite: np.ndarray       # value of `rush` with 自动打精英 on (zeros unless config.rush_elites)
    encounter: dict              # kind ("normal" / "elite") -> value with that enemy waiting on floor g
    boss: dict = field(default_factory=dict)   # floor -> [window, lambda, hp, boss hp]
    solve_seconds: float = 0.0

    def in_range(self, floor: int) -> bool:
        return 1 <= floor < self.config.max_floor

    def at(self, table: np.ndarray, floor: int, e: float) -> np.ndarray:
        """Values for every lambda at floor `floor`, effective HP `e`."""
        if not self.in_range(floor):
            return np.zeros(len(self.lambdas))
        return table[floor] @ interp_matrix(np.array([e]), self.grid)[0]

    def boss_at(self, floor: int, window: int, e: float, b: float) -> np.ndarray:
        if floor not in self.boss:
            return np.zeros(len(self.lambdas))
        w = min(window, BOSS_WINDOWS) - 1
        we = interp_matrix(np.array([e]), self.grid)[0]
        wb = interp_matrix(np.array([b]), self.boss_grid)[0]
        return np.einsum("lnb,n,b->l", self.boss[floor][w], we, wb)

    def price_index(self, values: np.ndarray, heal_units: float, smokes: int = 0) -> int:
        """Index of the prices minimising the dual bound V + lam * H + mu * S (heal units H, smokes S)."""
        return int(np.argmin(values + self.lambdas * max(0.0, heal_units) + self.smoke_prices * max(0, smokes)))


def grids(config: MdpConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(effective-HP grid, boss-HP grid, heal price, smoke price); the prices list every pair of the
    two configured grids. Boss HP starts just above 0 (nearly dead)."""
    grid = np.linspace(0.0, E_MAX, config.hp_points)
    boss_grid = np.concatenate([[0.02], np.linspace(0.0, 1.0, config.boss_points)[1:]])
    lam, mu = np.meshgrid(np.array(config.lambdas, dtype=float), np.array(config.smoke_prices, dtype=float),
                          indexing="ij")
    return grid, boss_grid, lam.ravel(), mu.ravel()


# --------------------------------------------------------------------------- kernels
def regular_kernel(build: Build, foe: dict, grid: np.ndarray, samples: int, rng: np.random.Generator,
                   rounds: int = REGULAR_ROUND_CAP) -> tuple[np.ndarray, np.ndarray]:
    """(kill, draw): P[i, j] of starting at effective HP grid[i] and landing at grid[j] after a regular
    fight that the enemy loses, or that hits the round cap (it retreats; a rush stops there).
    kill + draw rows sum to P(survive); row 0 (dead) is empty."""
    n = len(grid)
    start = np.repeat(np.arange(1, n), samples)
    e0 = grid[start]
    hp, shield = np.minimum(e0, 1.0) * build.hp, np.maximum(e0 - 1.0, 0.0) * build.hp
    hp, shield, ehp, *_ = fight(build, hp, shield, foe, boss=False, rounds=rounds, rng=rng)
    e1 = (hp + shield) / build.hp
    killed = (hp > 0) & (ehp <= 0)
    drawn = (hp > 0) & (ehp > 0)
    return (_deposit(start[killed], e1[killed], grid, n, 1.0 / samples),
            _deposit(start[drawn], e1[drawn], grid, n, 1.0 / samples))


def segment_kernel(build: Build, foe: dict, *, boss: bool, start_round: int, rounds: int, grid: np.ndarray,
                   enemy_grid: np.ndarray, samples: int, rng: np.random.Generator,
                   fresh: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One fight segment (boss window, elite half) from every (hp point, enemy-hp point) state, or with
    `fresh` from every hp point against an unhurt enemy. foe["hp"] is the enemy's full HP.
    Returns (cont, kill): cont[s, s'] = P(both still alive, landing at flattened state s' =
    hp index * len(enemy_grid) + enemy-hp index); kill[s, j] = P(the enemy dies and the player lands
    at hp point j), kill heal included. Rows of a dead start (hp point 0) are empty."""
    n, nb = len(grid), len(enemy_grid)
    states = n * nb
    rows = n if fresh else states
    start = np.repeat(np.arange(rows), samples)
    if fresh:
        ei, b0 = start, np.ones(len(start))
    else:
        ei, bi = np.divmod(start, nb)
        b0 = enemy_grid[bi]
    e0 = grid[ei]
    hp, shield = np.minimum(e0, 1.0) * build.hp, np.maximum(e0 - 1.0, 0.0) * build.hp
    hp, shield, ehp, *_ = fight(build, hp, shield, foe, boss=boss, ehp=b0 * foe["hp"],
                                start_round=start_round, rounds=rounds, rng=rng)
    alive = (hp > 0) & (ei > 0)
    won = alive & (ehp <= 0)
    going = alive & (ehp > 0)
    e1 = (hp + shield) / build.hp
    kill = _deposit(start[won], e1[won], grid, rows, 1.0 / samples)
    # bilinear deposit of the survivors: split on hp, then on enemy hp
    src = start[going] * states
    cont = np.zeros(rows * states)
    for ie, we in _split(e1[going], grid):
        for ib, wb in _split(ehp[going] / foe["hp"], enemy_grid):
            cont += np.bincount(src + ie * nb + ib, weights=we * wb / samples, minlength=rows * states)
    return cont.reshape(rows, states), kill


def boss_kernel(build: Build, foe: dict, window: int, grid: np.ndarray, boss_grid: np.ndarray,
                samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One boss window from every (hp point, boss-hp point).
    Returns (P, win): P[s, s'] over flattened (hp, boss hp) states while both live, win[s] = P(boss dies)."""
    cont, kill = segment_kernel(build, foe, boss=True, start_round=(window - 1) * BOSS_WINDOW + 1,
                                rounds=BOSS_WINDOW, grid=grid, enemy_grid=boss_grid, samples=samples, rng=rng)
    return cont, kill.sum(axis=1)


def elite_pause_kernel(build: Build, foe: dict, grid: np.ndarray, enemy_grid: np.ndarray, samples: int,
                       rng: np.random.Generator, rules: ServerRules) -> np.ndarray:
    """lands[s, j]: from the elite pause at flattened state s, P(the player survives the remaining
    rounds and ends at hp point j). The elite dying and the elite retreating at the cap lead to the same
    place (this floor's ready state), so only the player's HP matters."""
    cont, kill = segment_kernel(build, foe, boss=False, start_round=rules.elite_break + 1,
                                rounds=rules.elite_round_cap - rules.elite_break, grid=grid,
                                enemy_grid=enemy_grid, samples=samples, rng=rng)
    return cont.reshape(len(cont), len(grid), len(enemy_grid)).sum(axis=2) + kill


def elite_pause_value(lands: np.ndarray, ready: np.ndarray, heal: "_Closure", n_enemy: int,
                      death_cost: float = 0.0) -> np.ndarray:
    """Values at the elite pause, [price, hp point, elite-hp point]: heal with any item, then fight on.
    lands: elite_pause_kernel; ready: (prices, hp points) values of this floor's ready state;
    death_cost: what a death in the remaining rounds loses (gold objective)."""
    values = ready @ lands.T
    if death_cost:
        values = values - death_cost * death_of(lands)[None, :]
    return heal(values.reshape(len(ready), -1, n_enemy))


def _split(x: np.ndarray, grid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return [(lo, 1 - w), (hi, w)]


def flee_chance(build: Build, floor: int) -> float:
    """Server flee rate without gear or field modifiers, plus the rainbow 风隙羽标 bonus."""
    return min(1.0, min(0.90, max(0.15, 0.45 + (build.agi - 1.5 * floor) * 0.015)) + build.flee_bonus)


def flee_hit(build: Build, foe: dict) -> float:
    """Expected HP lost (fraction of max HP) to the parting blow of a failed flee (cannot be dodged)."""
    raw = foe["atk"] * 100 / (100 + 2.5 * build.eff_def) * (1 - build.reduction) * build.taken_mult
    return max(1.0, raw) / build.hp


# --------------------------------------------------------------------------- solver
class _Closure:
    """Heal closure V* = max(V, max_i V*(apply_i(e)) - lam * cost_i), vectorised over lambdas."""

    def __init__(self, items: list[HealItem], grid: np.ndarray, max_hp: float, lambdas: np.ndarray,
                 iterations: int):
        self.iterations = iterations
        if items:
            self.moves = np.stack([interp_matrix(it.apply(grid, max_hp), grid) for it in items])  # (I, N, N)
            self.costs = np.array([it.units(max_hp) for it in items])[:, None] * lambdas[None, :]  # (I, L)
        else:
            self.moves = None

    def __call__(self, base: np.ndarray) -> np.ndarray:
        """base: (L, N) or (L, N, B) - lambda first, hp second."""
        if self.moves is None:
            return base
        nl, n = base.shape[:2]
        rest = base.shape[2:]
        # hp axis first so one BLAS matmul per heal item covers every lambda and boss-hp column
        flat_base = np.moveaxis(base, 1, 0).reshape(n, -1)                       # (N, L*rest)
        width = flat_base.shape[1] // nl
        costs = np.repeat(self.costs, width, axis=1)[:, None, :]                 # (I, 1, L*rest)
        value = flat_base
        for _ in range(self.iterations):
            new = flat_base.copy()
            for move, cost in zip(self.moves, costs):                            # one BLAS call per item
                np.maximum(new, move @ value - cost, out=new)
            if np.allclose(new, value):
                break
            value = new
        return np.moveaxis(value.reshape((n, nl) + rest), 0, 1)


def solve(build: Build, map_id: str = "mall", config: MdpConfig = MdpConfig(),
          progress: Callable[[str], None] | None = None) -> ValueTables:
    """Backward induction from config.max_floor down to F1 for every heal price at once."""
    t0 = time.time()
    rng = np.random.default_rng(config.seed)
    grid, boss_grid, lambdas, smoke_prices = grids(config)
    n, nb, nl, top = len(grid), len(boss_grid), len(lambdas), config.max_floor
    max_hp = build.hp
    any_heal = _Closure(list(SHOP_HEALS.values()) + list(FIELD_HEALS.values()), grid, max_hp, lambdas,
                        config.heal_iterations)
    calm_heal = _Closure([h for h in SHOP_HEALS.values() if not h.combat_only] + list(FIELD_HEALS.values()),
                         grid, max_hp, lambdas, config.heal_iterations)
    full = interp_matrix(np.array([1.0]), grid)[0]                     # weights of e = 1.0
    low = (grid <= RUSH_STOP_HP)[None, :]                              # rush stops at HP <= 50%

    shape = (top + 2, nl, n)
    ready, step, rush, rush_elite = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    arrive = np.zeros(shape)
    encounter = {"normal": np.zeros(shape), "elite": np.zeros(shape)}
    boss_tables: dict[int, np.ndarray] = {}
    # rush continuation: mid[h][j-1] = value about to enter floor h with j floors left (not the first)
    mid = {top + 1: np.zeros((RUSH_FLOORS, nl, n)), top + 2: np.zeros((RUSH_FLOORS, nl, n))}
    mid_elite = {h: layers.copy() for h, layers in mid.items()}      # the same with 自动打精英 on
    # value of reaching the elite pause from a fresh elite on floor h, (L, start hp): heal, fight on
    pause_reached: dict[int, np.ndarray] = {}
    rules = config.rules
    kernels: dict[tuple[str, int], tuple] = {}

    def reference(floor: int) -> int:
        """Floor whose (strongest) enemy stands in for its kernel_stride block."""
        return min(top, (floor - 1) // config.kernel_stride * config.kernel_stride + config.kernel_stride)

    def kernel(kind: str, floor: int) -> tuple[np.ndarray, np.ndarray]:
        """(kill, draw) of a fight without a pause."""
        key = (kind, reference(floor))
        if key not in kernels:
            rounds = rules.elite_round_cap if kind == "elite" else REGULAR_ROUND_CAP
            kernels[key] = regular_kernel(build, foe_at(kind, key[1], map_id, build), grid, config.samples, rng, rounds)
        return kernels[key]

    def elite_kernels(floor: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """((cont, kill) of the rounds before the pause from a fresh elite, pause lands)."""
        key = ("elite_pause", reference(floor))
        if key not in kernels:
            foe = foe_at("elite", key[1], map_id, build)
            first = segment_kernel(build, foe, boss=False, start_round=1, rounds=rules.elite_break, grid=grid,
                                   enemy_grid=boss_grid, samples=config.samples, rng=rng, fresh=True)
            kernels[key] = first, elite_pause_kernel(build, foe, grid, boss_grid, config.pause_samples, rng, rules)
        return kernels[key]

    gain, risk = loot_profile(config, map_id, top)
    gold = config.gold

    def income(floor: int) -> np.ndarray:
        return (lambdas * field_heal_income(floor, max_hp))[:, None] + gain[floor]

    def died(h: int, *parts: np.ndarray) -> np.ndarray:
        """Value lost to deaths in a fight on floor h whose survivors land per `parts` (gold objective)."""
        return risk[h] * death_of(*parts)[None, :] if gold else 0.0

    def banked(values: np.ndarray, h: int) -> np.ndarray:
        """Gold objective: extracting is worth 0 more (it banks the bag); the dead state loses it."""
        if not gold:
            return values
        out = np.maximum(values, 0.0)
        out[..., 0] = -risk[h]
        return out

    def enter(h: int, after: np.ndarray, stop_low: bool, auto_elite: bool = False) -> np.ndarray:
        """Value of entering floor h during a rush; `after` = continuation once the floor is resolved.
        auto_elite: the rush fights an elite by itself (HP above 50%): to its pause under live rules,
        continuing on a kill; to the round cap under legacy rules, a draw halting the rush."""
        if is_boss_floor(h):
            return income(h) + boss_start[h]
        mix = floor_mix(h, rules)
        killed, drawn = kernel("normal", h)
        fought = after @ killed.T + ready[h] @ drawn.T - died(h, killed, drawn)   # a draw halts the rush there
        normal = np.where(low, encounter["normal"][h], fought) if stop_low else fought
        elite = encounter["elite"][h]
        if auto_elite:
            if rules.elite_break:
                (cont, kill), _ = elite_kernels(h)
                auto = after @ kill.T + pause_reached[h] - died(h, cont, kill)
            else:
                killed, drawn = kernel("elite", h)
                auto = after @ killed.T + ready[h] @ drawn.T - died(h, killed, drawn)
            elite = np.where(low, elite, auto)
        value = mix["normal"] * normal + mix["elite"] * elite + mix["choice"] * ready[h] + mix["loot"] * after
        return income(h) + value

    boss_start: dict[int, np.ndarray] = {}
    for g in range(top, 0, -1):
        # 1. rush started from a ready state on g (first floor: no pre-descent HP check)
        rush[g] = enter(g + 1, mid[g + 2][RUSH_FLOORS - 2], stop_low=True) if g + 1 <= top else 0.0
        if config.rush_elites and g + 1 <= top:
            rush_elite[g] = enter(g + 1, mid_elite[g + 2][RUSH_FLOORS - 2], stop_low=True, auto_elite=True)
        # 2. ready on g: step, rush (either way) or heal first
        step[g] = arrive[g + 1]
        ready[g] = banked(calm_heal(np.maximum(np.maximum(step[g], rush[g]), rush_elite[g])), g)
        # 3. encounters on g
        for kind in ("normal", "elite"):
            foe = enemy(kind, g, map_id)
            if kind == "elite" and rules.elite_break:
                (cont, kill), lands = elite_kernels(g)
                pause = elite_pause_value(lands, ready[g], any_heal, nb, risk[g])     # (L, N, Nb)
                if gold:
                    pause = np.maximum(pause, 0.0)                                  # extract at the pause
                    pause[:, 0, :] = -risk[g]                                       # dead there: the bag is lost
                pause_reached[g] = pause.reshape(nl, -1) @ cont.T
                fight_value = ready[g] @ kill.T + pause_reached[g] - died(g, cont, kill)
            else:
                killed, drawn = kernel(kind, g)
                fight_value = ready[g] @ (killed + drawn).T - died(g, killed, drawn)
            fail_at = interp_matrix(np.maximum(grid - flee_hit(build, foe), 0.0), grid)
            p = flee_chance(build, g)
            flee_value = p * ready[g] + (1 - p) * (fight_value @ fail_at.T)
            flee_value[:, 0] = -risk[g]
            smoke_value = ready[g] - smoke_prices[:, None]        # smoke bomb: leave unharmed, stay on g
            smoke_value[:, 0] = -risk[g]
            encounter[kind][g] = banked(any_heal(np.maximum(np.maximum(fight_value, flee_value), smoke_value)), g)
        # 4. boss on g
        if is_boss_floor(g):
            boss_tables[g] = _solve_boss(build, foe_at("boss", g, map_id, build), grid, boss_grid, lambdas,
                                         config.boss_value + ready[g] @ full, any_heal, config, rng,
                                         death_cost=risk[g])
            boss_start[g] = boss_tables[g][0][:, :, -1]
        # 5. single-step arrival on g
        if is_boss_floor(g):
            arrive[g] = income(g) + boss_start[g]
        else:
            mix = floor_mix(g, rules)
            arrive[g] = income(g) + (mix["normal"] * encounter["normal"][g] + mix["elite"] * encounter["elite"][g]
                                     + (mix["choice"] + mix["loot"]) * ready[g])
        if gold:
            arrive[g][:, 0] = -risk[g]
        # 6. mid-rush values entering g+1 (needs ready[g] for the pre-descent HP check)
        if g + 1 <= top:
            layers = np.zeros((RUSH_FLOORS, nl, n))
            for j in range(1, RUSH_FLOORS):
                after = ready[g + 1] if j == 1 else mid[g + 2][j - 2]
                layers[j - 1] = np.where(low, ready[g], enter(g + 1, after, stop_low=True))
            mid[g + 1] = layers
            mid.pop(g + 3, None)
            if config.rush_elites:
                layers = np.zeros((RUSH_FLOORS, nl, n))
                for j in range(1, RUSH_FLOORS):
                    after = ready[g + 1] if j == 1 else mid_elite[g + 2][j - 2]
                    layers[j - 1] = np.where(low, ready[g], enter(g + 1, after, stop_low=True, auto_elite=True))
                mid_elite[g + 1] = layers
                mid_elite.pop(g + 3, None)
            pause_reached.pop(g + 2, None)
        if progress and g % 200 == 0:
            progress(f"mdp: floor {g} ({time.time() - t0:.0f}s)")
    return ValueTables(config=config, build=build, map_id=map_id, grid=grid, boss_grid=boss_grid,
                       lambdas=lambdas, smoke_prices=smoke_prices, ready=f32(ready), step=f32(step),
                       rush=f32(rush), rush_elite=f32(rush_elite),
                       encounter={k: f32(v) for k, v in encounter.items()},
                       boss={f: f32(v) for f, v in boss_tables.items()}, solve_seconds=time.time() - t0)


def f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32)


def _solve_boss(build: Build, foe: dict, grid: np.ndarray, boss_grid: np.ndarray, lambdas: np.ndarray,
                win_value: np.ndarray, heal: _Closure, config: MdpConfig,
                rng: np.random.Generator, death_cost: float = 0.0) -> np.ndarray:
    """Values before each boss window: [window, lambda, hp, boss hp]. A kill is worth boss_value + ready
    (full HP). Gold objective: a death loses death_cost, and extracting during the fight is worth 0."""
    n, nb, nl = len(grid), len(boss_grid), len(lambdas)
    values = np.zeros((BOSS_WINDOWS, nl, n, nb))
    win_term = win_value[:, None]                                   # (L, 1)
    kernels = [boss_kernel(build, foe, w, grid, boss_grid, config.samples, rng) for w in range(1, BOSS_WINDOWS + 1)]

    def window_value(w: int, nxt: np.ndarray) -> np.ndarray:
        p, win = kernels[w]
        cont = nxt.reshape(nl, n * nb) @ p.T + win_term * win[None, :]
        if config.gold:
            cont = cont - death_cost * np.clip(1.0 - p.sum(axis=1) - win, 0.0, 1.0)[None, :]
        out = heal(cont.reshape(nl, n, nb))
        if config.gold:
            out = np.maximum(out, 0.0)
            out[:, 0, :] = -death_cost
        return out

    last = np.zeros((nl, n, nb))
    for _ in range(config.boss_iterations):
        new = window_value(BOSS_WINDOWS - 1, last)
        if np.allclose(new, last, atol=1e-6):
            break
        last = new
    values[BOSS_WINDOWS - 1] = last
    for w in range(BOSS_WINDOWS - 2, -1, -1):
        values[w] = window_value(w, values[w + 1])
    return values


def tables_key(build: Build, map_id: str, config: MdpConfig) -> str:
    """Stable cache key of a solve."""
    text = repr((MODEL_VERSION, map_id, astuple(build), astuple(config)))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def save_tables(tables: ValueTables, path: str) -> None:
    arrays = {"ready": tables.ready, "step": tables.step, "rush": tables.rush, "rush_elite": tables.rush_elite,
              "enc_normal": tables.encounter["normal"], "enc_elite": tables.encounter["elite"],
              "boss_floors": np.array(sorted(tables.boss), dtype=int)}
    if tables.boss:
        arrays["boss"] = np.stack([tables.boss[f] for f in sorted(tables.boss)])
    # Write next to the target, then rename: a process stopped mid-write (Ctrl+C while a background solve
    # saves) leaves a stray .part file, never a truncated cache entry that a later run would try to read.
    part = f"{path}.{os.getpid()}.part"
    try:
        with open(part, "wb") as fh:
            np.savez_compressed(fh, **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                                       for k, v in arrays.items()})
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def load_tables(path: str, build: Build, map_id: str, config: MdpConfig) -> ValueTables:
    grid, boss_grid, lambdas, smoke_prices = grids(config)
    with np.load(path) as data:
        floors = data["boss_floors"].tolist()
        boss = {f: data["boss"][i] for i, f in enumerate(floors)} if floors else {}
        return ValueTables(config=config, build=build, map_id=map_id, grid=grid, boss_grid=boss_grid,
                           lambdas=lambdas, smoke_prices=smoke_prices, ready=data["ready"], step=data["step"],
                           rush=data["rush"], rush_elite=data["rush_elite"],
                           encounter={"normal": data["enc_normal"], "elite": data["enc_elite"]},
                           boss=boss)


def with_calibration(build: Build, k_player: float, k_enemy: float) -> Build:
    """Apply the online damage calibration the same way combat.estimate does."""
    return replace(build, dealt_mult=build.dealt_mult * k_player, taken_mult=build.taken_mult * k_enemy)
