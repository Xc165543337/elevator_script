"""Offline game model: exact monster growth + vectorised combat + whole-run simulation.

Used to compare gear builds and temp-gear (临装) choices without touching the live game.
All rules come from the in-game formula sheet (docs/formulas_zh.txt); monster stats were
verified against live data (e.g. the F1030 Mall boss: 4,067 HP predicted and observed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

MAP_MULT = {"mall": 1.0, "subway": 1.6, "abyss": 2.45, "auction": 3.3, "tide": 3.85, "zenith": 4.45}
_GROWTH = {  # (a, b, c) per stat: value = a + b*x + c*z
    "normal": {"hp": (40, 1.15, 4.5), "atk": (10.8, 0.13, 0.88), "def": (2.7, 0.045, 0.16), "agi": (7, 0.09, 0.58)},
    "elite": {"hp": (74, 1.9, 11.4), "atk": (15.2, 0.16, 1.16), "def": (3.1, 0.07, 0.24), "agi": (8, 0.11, 0.74)},
    "boss": {"hp": (198, 2.6, 7.1), "atk": (30.5, 0.21, 0.58), "def": (4.4, 0.095, 0.33), "agi": (9, 0.12, 0.8)},
}
BOSS_WINDOW, BOSS_HEAVY = 4, 1.26
# Approximate floor mix observed in the Mall (per non-boss floor).
P_MOB, P_ELITE = 0.23, 0.14


def boss_floors(limit: int) -> list[int]:
    floors = [8, 18, 30]
    while floors[-1] + 20 <= limit:
        floors.append(floors[-1] + 20)
    return [f for f in floors if f <= limit]


def boss_index(floor: int) -> int:
    return {8: 1, 18: 2, 30: 3}.get(floor) or 3 + (floor - 30) // 20


def is_boss_floor(floor: int) -> bool:
    return floor in (8, 18, 30) or (floor > 30 and (floor - 30) % 20 == 0)


@lru_cache(maxsize=None)
def enemy(kind: str, floor: int, map_id: str = "mall") -> dict:
    x = floor - 1
    z = 6 * math.log(1 + x / 6)
    g = _GROWTH[kind]
    stat = {k: a + b * x + c * z for k, (a, b, c) in g.items()}
    m = MAP_MULT[map_id]
    out = {"hp": stat["hp"] * m, "atk": stat["atk"] * m, "def": stat["def"] * m, "agi": min(55.0, stat["agi"])}
    if kind == "boss":
        n = boss_index(floor)
        out["hp"] *= 1 + 0.055 * math.log2(n)
        out["atk"] *= 1 + 0.033 * math.log2(n)
    return {k: math.floor(v) for k, v in out.items()}


@dataclass(frozen=True)
class Build:
    """Everything that matters in combat. Rates are fractions (0.26 = 26%)."""
    hp: float
    atk: float
    defense: float
    agi: float
    luk: float
    pierce: float = 0.0
    leech: float = 0.0
    combo: float = 0.0
    heavy: float = 0.0          # 重击: first-round main attack bonus
    berserk: float = 0.0        # 越战越勇: per-round attack bonus (cap 75%)
    vampire: float = 0.0        # 嗜血: heal fraction of max HP per kill
    thorns: float = 0.0         # 反甲刺猬: reflect fraction (cap 70%)
    boss_bonus: float = 0.0     # damage vs bosses (cap 120%)
    crit_mult: float = 1.6
    crit_bonus: float = 0.0     # temp-gear crit modifier
    desperate: bool = False     # 孤注一掷
    last_stand: bool = False    # 不屈
    start_shield: float = 0.0
    dealt_mult: float = 1.0     # temp gear two-way multipliers (策划的平衡性调整: x2 / x2)
    taken_mult: float = 1.0
    reduction: float = 0.0      # 减伤 (negative = more damage taken)
    def_pct: float = 0.0        # temp-gear defence %
    atk_pct: float = 0.0        # temp-gear / run attack %
    label: str = ""

    @property
    def eff_def(self) -> float:
        return max(0.0, self.defense * (1 + self.def_pct))

    @property
    def eff_atk(self) -> float:
        return self.atk * max(0.25, 1 + self.atk_pct)

    @property
    def dodge(self) -> float:
        return min(0.52, max(0.0, min(0.45, 0.03 + 0.004 * self.agi)))

    def crit(self, low_hp: bool = False) -> float:
        base = min(0.60, (0.05 + 0.005 * max(0.0, self.luk)) * (2 if (low_hp and self.desperate) else 1))
        return min(0.65, max(0.0, base + self.crit_bonus))

    def with_temp_gear(self, *items: str) -> "Build":
        b = self
        for item in items:
            b = replace(b, **TEMP_GEAR[item](b))
        return b


# Wearable temp gear (临装) available in the Mall pool, as Build modifiers.
TEMP_GEAR = {
    "balance": lambda b: {"dealt_mult": b.dealt_mult * 2, "taken_mult": b.taken_mult * 2},   # 策划的平衡性调整
    "apron": lambda b: {"def_pct": b.def_pct + 0.30, "agi": b.agi - 3},                      # 厚帆布围裙
    "harness": lambda b: {"def_pct": b.def_pct + 0.15, "agi": b.agi + 5},                    # 通勤束带
    "spotlight": lambda b: {"crit_bonus": b.crit_bonus + 0.13, "reduction": b.reduction - 0.10},  # 促销聚光灯
}


def fight(build: Build, hp: np.ndarray, shield: np.ndarray, foe: dict, *, boss: bool,
          ehp: np.ndarray | None = None, start_round: int = 1, rounds: int = 80,
          last_stand: np.ndarray | None = None, rng: np.random.Generator) -> tuple:
    """Vectorised fight over N parallel samples. Returns (hp, shield, ehp, last_stand, rounds_used).

    For bosses pass rounds=4-((start_round-1)%4) to stop at the end of a window.
    Samples with hp<=0 are dead; ehp<=0 means the enemy died (kill heal already applied).
    """
    n = hp.shape[0]
    hp, shield = hp.astype(float).copy(), shield.astype(float).copy()
    ehp = np.full(n, float(foe["hp"])) if ehp is None else ehp.astype(float).copy()
    ls = np.full(n, build.last_stand) if last_stand is None else last_stand.copy()
    max_hp = build.hp
    edodge = min(0.45, 0.03 + 0.004 * foe["agi"])
    pdodge = build.dodge
    p_raw = build.eff_atk * 100 / (100 + 2.5 * foe["def"] * (1 - min(0.75, build.pierce)))
    e_raw = foe["atk"] * 100 / (100 + 2.5 * build.eff_def) * (1 - build.reduction) * build.taken_mult
    boss_mult = 1 + min(1.2, build.boss_bonus) if boss else 1.0
    player_first = build.agi >= foe["agi"]
    used = np.zeros(n)
    active = (hp > 0) & (ehp > 0)
    for i in range(rounds):
        if not active.any():
            break
        t = start_round + i
        used[active] += 1

        def player_attack(mask):
            nonlocal hp
            low = hp < 0.3 * max_hp
            mult = (1 + min(0.75, (t - 1) * build.berserk)) * boss_mult
            if t == 1:
                mult *= 1 + build.heavy
            atk_mult = np.where(low & build.desperate, 1.75, 1.0)
            crit_p = np.where(low, build.crit(True), build.crit(False))
            hit = rng.random(n) >= edodge
            crit = rng.random(n) < crit_p
            dmg = p_raw * mult * atk_mult * rng.uniform(0.9, 1.1, n) * np.where(crit, build.crit_mult, 1.0)
            dmg = np.floor(dmg * build.dealt_mult) * hit
            if build.combo > 0:
                extra = (rng.random(n) < min(0.7, build.combo)) & (rng.random(n) >= edodge)
                dmg = dmg + np.floor(0.6 * p_raw * mult * atk_mult * rng.uniform(0.9, 1.1, n) * build.dealt_mult) * extra
            dmg = np.maximum(dmg, hit * 1.0) * mask
            dealt = np.minimum(dmg, np.maximum(ehp, 0))
            np.subtract(ehp, dmg, out=ehp)
            if build.leech > 0:
                heal = np.minimum(np.floor(dealt * min(0.42, build.leech)), max(1.0, 0.10 * max_hp))
                hp = np.minimum(max_hp, hp + heal * mask)

        def enemy_attack(mask):
            nonlocal hp, shield, ehp, ls
            hit = (rng.random(n) >= pdodge) & mask & (ehp > 0)
            dmg = e_raw * rng.uniform(0.9, 1.1, n)
            if boss and t % BOSS_WINDOW == 0:
                dmg = dmg * BOSS_HEAVY
            dmg = np.floor(np.maximum(dmg, 1)) * hit
            absorbed = np.minimum(shield, dmg)
            shield -= absorbed
            hp -= dmg - absorbed
            if build.thorns > 0:
                ehp -= np.floor(dmg * min(0.7, build.thorns)) * (hp > 0)
            saved = ls & (hp <= 0)
            hp = np.where(saved, 1.0, hp)
            ls = ls & ~saved

        for who in ((player_attack, enemy_attack) if player_first else (enemy_attack, player_attack)):
            who(active & (hp > 0) & (ehp > 0))
        active = (hp > 0) & (ehp > 0)
    killed = (ehp <= 0) & (hp > 0)
    if build.vampire > 0:
        hp = np.where(killed, np.minimum(max_hp, hp + build.vampire * max_hp), hp)
    return hp, shield, ehp, ls, used


@dataclass
class RunResult:
    bosses: float
    floor: float
    heals_used: float
    boss_windows_per_boss: float
    p_reach: dict


def simulate_runs(build: Build, *, runs: int = 300, heals: int = 30, max_floor: int = 2000,
                  free_heal_rate: float = 0.12, free_heal: float = 0.25, risk: float = 0.02,
                  atk_gain_per_floor: float = 0.20 / 150, event_atk_cap: float = 0.60,
                  adrenaline: int = 0, map_id: str = "mall", seed: int = 0) -> RunResult:
    """Simulate full runs. Heal policy: full heal (战地手术匣) before an elite fight or boss window
    when current HP is below the HP needed to survive it with probability 1-risk.

    adrenaline: shots used on F1 (+25% run attack each, max 2, each costs 10% current HP).
    Event attack buffs (强行超频 etc.) accrue at atk_gain_per_floor up to event_atk_cap."""
    rng = np.random.default_rng(seed)
    shots = min(2, adrenaline)
    build = replace(build, atk_pct=build.atk_pct + 0.25 * shots)
    hp = np.full(runs, float(build.hp) * 0.9 ** shots)
    shield = np.full(runs, float(build.start_shield))
    ls = np.full(runs, build.last_stand)
    heal_left = np.full(runs, heals)
    bosses = np.zeros(runs)
    windows = np.zeros(runs)
    alive = np.ones(runs, bool)
    floor_reached = np.zeros(runs)
    reach = {}
    base_build = build
    for floor in range(1, max_floor + 1):
        if not alive.any():
            break
        floor_reached[alive] = floor
        if floor % 10 == 1:  # run-long attack buffs from events (e.g. 强行超频 +20%), on a 10-floor grid
            gained = min(event_atk_cap, atk_gain_per_floor * (floor - 1))
            build = replace(base_build, atk_pct=base_build.atk_pct + gained)
        # free field heals (bandages etc.) used when below 75%
        got = alive & (rng.random(runs) < free_heal_rate) & (hp < 0.75 * build.hp)
        hp = np.where(got, np.minimum(build.hp, hp + free_heal * build.hp), hp)
        if is_boss_floor(floor):
            foe = enemy("boss", floor, map_id)
            need = _need_hp(build, "boss", floor, map_id, risk)
            ehp = np.full(runs, float(foe["hp"]))
            fighting = alive.copy()
            rnd = 1
            for _ in range(60):  # windows
                if not fighting.any():
                    break
                heal_now = fighting & (hp < need) & (heal_left > 0)
                hp = np.where(heal_now, build.hp, hp)
                heal_left -= heal_now
                windows += fighting
                hp2, shield2, ehp2, ls2, _ = fight(build, hp, shield, foe, boss=True, ehp=ehp, start_round=rnd,
                                                   rounds=BOSS_WINDOW, last_stand=ls, rng=rng)
                hp = np.where(fighting, hp2, hp)
                shield = np.where(fighting, shield2, shield)
                ehp = np.where(fighting, ehp2, ehp)
                ls = np.where(fighting, ls2, ls)
                rnd += BOSS_WINDOW
                dead = fighting & (hp <= 0)
                won = fighting & (ehp <= 0) & (hp > 0)
                alive &= ~dead
                bosses += won
                hp = np.where(won, build.hp, hp)  # boss kill: full heal
                fighting &= ~(dead | won)
            alive &= ~fighting  # boss never died within 60 windows -> treat as stuck
        else:
            roll = rng.random(runs)
            for kind, mask in (("normal", roll < P_MOB), ("elite", (roll >= P_MOB) & (roll < P_MOB + P_ELITE))):
                mask &= alive
                if not mask.any():
                    continue
                need = _need_hp(build, kind, floor, map_id, risk)
                heal_now = mask & (hp < need) & (heal_left > 0)
                hp = np.where(heal_now, build.hp, hp)
                heal_left -= heal_now
                hp2, shield2, _, ls2, _ = fight(build, hp, shield, enemy(kind, floor, map_id), boss=False,
                                                last_stand=ls, rng=rng)
                hp = np.where(mask, hp2, hp)
                shield = np.where(mask, shield2, shield)
                ls = np.where(mask, ls2, ls)
                alive &= ~(mask & (hp <= 0))
        if floor % 100 == 0:
            reach[floor] = float(alive.mean())
    return RunResult(bosses=float(bosses.mean()), floor=float(floor_reached.mean()),
                     heals_used=float((heals - heal_left).mean()),
                     boss_windows_per_boss=float(windows.sum() / max(1.0, bosses.sum())), p_reach=reach)


@lru_cache(maxsize=200_000)
def _need_hp(build: Build, kind: str, floor: int, map_id: str, risk: float) -> float:
    """Smallest starting HP (on a 5% grid) that survives this fight (boss: one window) w.p. >= 1-risk."""
    bucket = floor - floor % 10 if kind != "boss" else floor  # coarse cache for regular fights
    foe = enemy(kind, max(1, bucket), map_id)
    rng = np.random.default_rng(bucket)
    n = 400
    for frac in np.arange(0.05, 1.0001, 0.05):
        hp = np.full(n, frac * build.hp)
        out, *_ = fight(build, hp, np.zeros(n), foe, boss=(kind == "boss"),
                        rounds=BOSS_WINDOW if kind == "boss" else 80, rng=rng)
        if (out <= 0).mean() <= risk:
            return frac * build.hp
    return build.hp + 1  # cannot be made safe: heal anyway
