"""Markov decision process for boss farming, solved by backward induction.

Objective: maximise the expected number of formal boss kills of one run (each kill is an independent
0.5% rainbow roll). Death and extraction both end the run, so neither carries value of its own.

State (per floor g): effective HP `e` = (HP + shield) / max HP on a grid over [0, E_MAX], plus the
decision point:
  * ready      - nothing pending: heal (non-combat items), step one floor, or rush;
  * encounter  - a normal monster or elite is waiting: heal (any item), fight, or flee;
  * boss       - before boss window w (1..4, window 4+ is stationary) with boss HP fraction b.

Healing stock is a run-long budget shared by every decision. Tracking it exactly would multiply the
state space by the inventory, so the budget constraint is relaxed with a Lagrange multiplier: for a
price `lam` (bosses per max-HP unit of healing) each heal costs `lam * units`, and found field heals
earn `lam * units`. Tables are solved for a grid of prices at once. At decision time the price is
chosen by the dual bound  V(s, H) ~= min_lam V_lam(s) + lam * H  for the live heal stock H, and the
action is the best one under that price. This is re-evaluated at every decision, so the price rises
as the stock runs down.

Transitions come from the same combat model as everything else (model.fight, Monte Carlo per grid
point); encounter odds from model.floor_mix. Events, loot and relics leave HP unchanged here (the live
policy still handles their content).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import astuple, dataclass, field, replace
from typing import Callable

import numpy as np

from model import BOSS_WINDOW, REGULAR_ROUND_CAP, Build, enemy, fight, floor_mix, is_boss_floor

E_MAX = 1.25                 # effective HP cap: full HP plus the 25% shield of a barrier serum
BOSS_WINDOWS = 4             # windows 1-3 differ (first-round bonus, berserk ramp); 4+ is stationary
RUSH_FLOORS = 10             # floors a rush descends at most (server setting)
RUSH_STOP_HP = 0.5           # the server stops a rush at HP <= 50%
FIELD_DROP_CHANCE = 0.52     # server FIELD_CHANCE; guaranteed on F1, boss floors and the floor before
MODEL_VERSION = 1            # bump when the dynamics change, to invalidate cached tables


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
    max_floor: int = 2600          # values beyond are 0 (no run has come close)
    hp_points: int = 51            # effective-HP grid points over [0, E_MAX]
    boss_points: int = 11          # boss-HP grid points over (0, 1]
    samples: int = 256             # Monte Carlo fights per grid point
    kernel_stride: int = 4         # regular-fight kernels are shared by this many floors (strongest foe)
    boss_iterations: int = 60      # value iteration of the stationary boss window (server cap: 60 windows)
    heal_iterations: int = 8       # heal chains considered per decision (closure depth)
    lambdas: tuple = (0.0, 0.001, 0.003, 0.006, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25,
                      0.4, 0.7, 1.2, 2.0, 4.0, 10.0)
    seed: int = 0


@dataclass
class ValueTables:
    """Solved values, indexed [floor, lambda, hp point] (boss: [window, lambda, hp, boss hp])."""
    config: MdpConfig
    build: Build
    map_id: str
    grid: np.ndarray
    boss_grid: np.ndarray
    lambdas: np.ndarray
    ready: np.ndarray            # value at a ready state on floor g
    step: np.ndarray             # value of `down` from a ready state on floor g
    rush: np.ndarray             # value of `rush` from a ready state on floor g
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

    def price_index(self, values: np.ndarray, heal_units: float) -> int:
        """Index of the heal price minimising the dual bound V_lam + lam * H."""
        return int(np.argmin(values + self.lambdas * max(0.0, heal_units)))


def grids(config: MdpConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(effective-HP grid, boss-HP grid, heal prices). Boss HP starts just above 0 (nearly dead)."""
    grid = np.linspace(0.0, E_MAX, config.hp_points)
    boss_grid = np.concatenate([[0.02], np.linspace(0.0, 1.0, config.boss_points)[1:]])
    return grid, boss_grid, np.array(config.lambdas, dtype=float)


# --------------------------------------------------------------------------- kernels
def regular_kernel(build: Build, foe: dict, grid: np.ndarray, samples: int,
                   rng: np.random.Generator) -> np.ndarray:
    """P[i, j]: start at effective HP grid[i], fight a regular enemy to the end (kill, draw or death),
    land at grid[j]. Rows sum to P(survive); row 0 (dead) is empty."""
    n = len(grid)
    start = np.repeat(np.arange(1, n), samples)
    e0 = grid[start]
    hp, shield = np.minimum(e0, 1.0) * build.hp, np.maximum(e0 - 1.0, 0.0) * build.hp
    hp, shield, *_ = fight(build, hp, shield, foe, boss=False, rounds=REGULAR_ROUND_CAP, rng=rng)
    alive = hp > 0
    e1 = (hp + shield) / build.hp
    return _deposit(start[alive], e1[alive], grid, n, 1.0 / samples)


def boss_kernel(build: Build, foe: dict, window: int, grid: np.ndarray, boss_grid: np.ndarray,
                samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One boss window from every (hp point, boss-hp point).
    Returns (P, win): P[s, s'] over flattened (hp, boss hp) states while both live, win[s] = P(boss dies)."""
    n, nb = len(grid), len(boss_grid)
    states = n * nb
    start = np.repeat(np.arange(states), samples)
    ei, bi = np.divmod(start, nb)
    e0, b0 = grid[ei], boss_grid[bi]
    hp, shield = np.minimum(e0, 1.0) * build.hp, np.maximum(e0 - 1.0, 0.0) * build.hp
    first_round = (window - 1) * BOSS_WINDOW + 1
    hp, shield, ehp, *_ = fight(build, hp, shield, foe, boss=True, ehp=b0 * foe["hp"],
                                start_round=first_round, rounds=BOSS_WINDOW, rng=rng)
    alive = hp > 0
    won = alive & (ehp <= 0) & (ei > 0)
    going = alive & (ehp > 0) & (ei > 0)
    win = np.bincount(start[won], minlength=states) / samples
    e1 = (hp + shield)[going] / build.hp
    b1 = ehp[going] / foe["hp"]
    # bilinear deposit: split on hp, then on boss hp
    src = start[going] * states
    p = np.zeros(states * states)
    for ie, we in _split(e1, grid):
        for ib, wb in _split(b1, boss_grid):
            p += np.bincount(src + ie * nb + ib, weights=we * wb / samples, minlength=states * states)
    return p.reshape(states, states), win


def _split(x: np.ndarray, grid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="right"), 1, len(grid) - 1)
    lo = hi - 1
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return [(lo, 1 - w), (hi, w)]


def flee_chance(build: Build, floor: int) -> float:
    """Server flee rate without perks or field modifiers."""
    return min(0.90, max(0.15, 0.45 + (build.agi - 1.5 * floor) * 0.015))


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
            moved = (self.moves @ value - costs).max(axis=0)                      # (N, L*rest)
            new = np.maximum(flat_base, moved)
            if np.allclose(new, value):
                break
            value = new
        return np.moveaxis(value.reshape((n, nl) + rest), 0, 1)


def solve(build: Build, map_id: str = "mall", config: MdpConfig = MdpConfig(),
          progress: Callable[[str], None] | None = None) -> ValueTables:
    """Backward induction from config.max_floor down to F1 for every heal price at once."""
    t0 = time.time()
    rng = np.random.default_rng(config.seed)
    grid, boss_grid, lambdas = grids(config)
    n, nb, nl, top = len(grid), len(boss_grid), len(lambdas), config.max_floor
    max_hp = build.hp
    any_heal = _Closure(list(SHOP_HEALS.values()) + list(FIELD_HEALS.values()), grid, max_hp, lambdas,
                        config.heal_iterations)
    calm_heal = _Closure([h for h in SHOP_HEALS.values() if not h.combat_only] + list(FIELD_HEALS.values()),
                         grid, max_hp, lambdas, config.heal_iterations)
    full = interp_matrix(np.array([1.0]), grid)[0]                     # weights of e = 1.0
    low = (grid <= RUSH_STOP_HP)[None, :]                              # rush stops at HP <= 50%

    shape = (top + 2, nl, n)
    ready, step, rush = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    arrive = np.zeros(shape)
    encounter = {"normal": np.zeros(shape), "elite": np.zeros(shape)}
    boss_tables: dict[int, np.ndarray] = {}
    # rush continuation: mid[h][j-1] = value about to enter floor h with j floors left (not the first)
    mid = {top + 1: np.zeros((RUSH_FLOORS, nl, n)), top + 2: np.zeros((RUSH_FLOORS, nl, n))}
    kernels: dict[tuple[str, int], np.ndarray] = {}

    def kernel(kind: str, floor: int) -> np.ndarray:
        ref = min(top, (floor - 1) // config.kernel_stride * config.kernel_stride + config.kernel_stride)
        key = (kind, ref)
        if key not in kernels:
            kernels[key] = regular_kernel(build, enemy(kind, ref, map_id), grid, config.samples, rng)
        return kernels[key]

    def income(floor: int) -> np.ndarray:
        return (lambdas * field_heal_income(floor, max_hp))[:, None]

    def enter(h: int, after: np.ndarray, stop_low: bool) -> np.ndarray:
        """Value of entering floor h during a rush; `after` = continuation once the floor is resolved."""
        if is_boss_floor(h):
            return income(h) + boss_start[h]
        mix = floor_mix(h)
        fought = after @ kernel("normal", h).T
        normal = np.where(low, encounter["normal"][h], fought) if stop_low else fought
        value = (mix["normal"] * normal + mix["elite"] * encounter["elite"][h]
                 + mix["choice"] * ready[h] + mix["loot"] * after)
        return income(h) + value

    boss_start: dict[int, np.ndarray] = {}
    for g in range(top, 0, -1):
        # 1. rush started from a ready state on g (first floor: no pre-descent HP check)
        rush[g] = enter(g + 1, mid[g + 2][RUSH_FLOORS - 2], stop_low=True) if g + 1 <= top else 0.0
        # 2. ready on g: step, rush or heal first
        step[g] = arrive[g + 1]
        ready[g] = calm_heal(np.maximum(step[g], rush[g]))
        # 3. encounters on g
        for kind in ("normal", "elite"):
            foe = enemy(kind, g, map_id)
            fight_value = ready[g] @ kernel(kind, g).T
            fail_at = interp_matrix(np.maximum(grid - flee_hit(build, foe), 0.0), grid)
            p = flee_chance(build, g)
            flee_value = p * ready[g] + (1 - p) * (fight_value @ fail_at.T)
            flee_value[:, 0] = 0.0
            encounter[kind][g] = any_heal(np.maximum(fight_value, flee_value))
        # 4. boss on g
        if is_boss_floor(g):
            boss_tables[g] = _solve_boss(build, enemy("boss", g, map_id), grid, boss_grid, lambdas,
                                         1.0 + ready[g] @ full, any_heal, config, rng)
            boss_start[g] = boss_tables[g][0][:, :, -1]
        # 5. single-step arrival on g
        if is_boss_floor(g):
            arrive[g] = income(g) + boss_start[g]
        else:
            mix = floor_mix(g)
            arrive[g] = income(g) + (mix["normal"] * encounter["normal"][g] + mix["elite"] * encounter["elite"][g]
                                     + (mix["choice"] + mix["loot"]) * ready[g])
        # 6. mid-rush values entering g+1 (needs ready[g] for the pre-descent HP check)
        if g + 1 <= top:
            layers = np.zeros((RUSH_FLOORS, nl, n))
            for j in range(1, RUSH_FLOORS):
                after = ready[g + 1] if j == 1 else mid[g + 2][j - 2]
                layers[j - 1] = np.where(low, ready[g], enter(g + 1, after, stop_low=True))
            mid[g + 1] = layers
            mid.pop(g + 3, None)
        if progress and g % 200 == 0:
            progress(f"mdp: floor {g} ({time.time() - t0:.0f}s)")
    return ValueTables(config=config, build=build, map_id=map_id, grid=grid, boss_grid=boss_grid,
                       lambdas=lambdas, ready=ready, step=step, rush=rush, encounter=encounter,
                       boss=boss_tables, solve_seconds=time.time() - t0)


def _solve_boss(build: Build, foe: dict, grid: np.ndarray, boss_grid: np.ndarray, lambdas: np.ndarray,
                win_value: np.ndarray, heal: _Closure, config: MdpConfig,
                rng: np.random.Generator) -> np.ndarray:
    """Values before each boss window: [window, lambda, hp, boss hp]. A kill is worth 1 + ready(full HP)."""
    n, nb, nl = len(grid), len(boss_grid), len(lambdas)
    values = np.zeros((BOSS_WINDOWS, nl, n, nb))
    win_term = win_value[:, None]                                   # (L, 1)
    kernels = [boss_kernel(build, foe, w, grid, boss_grid, config.samples, rng) for w in range(1, BOSS_WINDOWS + 1)]

    def window_value(w: int, nxt: np.ndarray) -> np.ndarray:
        p, win = kernels[w]
        cont = nxt.reshape(nl, n * nb) @ p.T + win_term * win[None, :]
        return heal(cont.reshape(nl, n, nb))

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
    arrays = {"ready": tables.ready, "step": tables.step, "rush": tables.rush,
              "enc_normal": tables.encounter["normal"], "enc_elite": tables.encounter["elite"],
              "boss_floors": np.array(sorted(tables.boss), dtype=int)}
    if tables.boss:
        arrays["boss"] = np.stack([tables.boss[f] for f in sorted(tables.boss)])
    np.savez_compressed(path, **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in arrays.items()})


def load_tables(path: str, build: Build, map_id: str, config: MdpConfig) -> ValueTables:
    grid, boss_grid, lambdas = grids(config)
    with np.load(path) as data:
        floors = data["boss_floors"].tolist()
        boss = {f: data["boss"][i].astype(float) for i, f in enumerate(floors)} if floors else {}
        return ValueTables(config=config, build=build, map_id=map_id,
                           grid=grid, boss_grid=boss_grid, lambdas=lambdas,
                           ready=data["ready"].astype(float), step=data["step"].astype(float),
                           rush=data["rush"].astype(float),
                           encounter={"normal": data["enc_normal"].astype(float),
                                      "elite": data["enc_elite"].astype(float)},
                           boss=boss)


def with_calibration(build: Build, k_player: float, k_enemy: float) -> Build:
    """Apply the online damage calibration the same way combat.estimate does."""
    return replace(build, dealt_mult=build.dealt_mult * k_player, taken_mult=build.taken_mult * k_enemy)
