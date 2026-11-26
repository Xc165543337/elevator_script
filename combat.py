"""Live fight estimation: builds a model.Build from the in-run state and Monte Carlos the
current fight with the vectorised engine in model.py.

Two calibration factors (observed / formula damage, learned online from battle logs) absorb
anything the formulas do not expose. Logged damage is divided by the run's two-way damage
multiplier (策划的平衡性调整) before it is compared with the formula.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace

import numpy as np

import heirlooms
from model import (BEAT_EVERY, BEAT_TAKEN, BOSS_WINDOW, LIVE_RULES, Build, ServerRules, fight, fight_segment,
                   floor_mix, foe_at)

BERSERK_CAP = 0.75


def raw_hit(atk: float, defense: float) -> float:
    return atk * 100.0 / (100.0 + 2.5 * max(0.0, defense))


_SEG_ROUND = re.compile(r"^R(\d+)：")
_SEG_PLAYER = re.compile(r"^你打 (\d+)")
_SEG_ENEMY = re.compile(r"^怪打 (\d+)（你剩\d+，盾挡(\d+)）")


def parse_battle_log(log: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Extract non-crit player hits and normal enemy hits (incl. shield-absorbed part)."""
    out = {"player": [], "enemy": []}
    for line in log:
        m = _SEG_ROUND.match(line)
        rnd = int(m.group(1)) if m else 0
        for seg in _SEG_ROUND.sub("", line).split(" ｜ "):
            if (pm := _SEG_PLAYER.match(seg)):
                out["player"].append((rnd, float(pm.group(1))))
            elif (em := _SEG_ENEMY.match(seg)):
                out["enemy"].append((rnd, float(em.group(1)) + float(em.group(2))))
    return out


@dataclass
class Calibration:
    """Running estimates of observed / predicted damage, persisted between sessions."""
    k_player: float = 1.0
    k_enemy: float = 1.0
    n_player: int = 0
    n_enemy: int = 0
    path: str = field(default="calibration.json", repr=False)
    alpha: float = field(default=0.05, repr=False)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(path=path, **{k: data[k] for k in ("k_player", "k_enemy", "n_player", "n_enemy") if k in data})
        return cls(path=path)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"k_player": self.k_player, "k_enemy": self.k_enemy,
                       "n_player": self.n_player, "n_enemy": self.n_enemy}, fh, indent=1)

    def conservative(self) -> tuple[float, float]:
        """Factors used for decisions. The formulas were verified exact; learned values below/above 1
        mostly reflect temporary buffs (damage reduction, shields), so never let them make a fight
        look easier than the formula says."""
        return min(1.0, self.k_player), max(1.0, self.k_enemy)

    def _update(self, attr: str, count_attr: str, ratio: float) -> None:
        n = getattr(self, count_attr) + 1
        weight = max(self.alpha, 1.0 / n)
        setattr(self, attr, (1 - weight) * getattr(self, attr) + weight * ratio)
        setattr(self, count_attr, n)

    def observe(self, build: Build, enemy: dict, log: list[str]) -> None:
        """Update factors from one battle log, given the Build that was used to predict it."""
        hits = parse_battle_log(log)
        boss_mult = 1 + min(1.2, build.boss_bonus) if enemy.get("boss") else 1.0
        p_pred = raw_hit(build.eff_atk, enemy["def"] * (1 - min(build.pierce_cap, build.pierce))) * build.dealt_mult * boss_mult
        e_pred = raw_hit(enemy["atk"], build.eff_def) * build.taken_mult * (1 - build.reduction)
        for rnd, dmg in hits["player"]:
            if rnd > 1 and p_pred > 0:  # round 1 carries one-off bonuses
                berserk = 1 + min(BERSERK_CAP, (rnd - 1) * build.berserk)
                self._update("k_player", "n_player", dmg / (p_pred * berserk))
        for rnd, dmg in hits["enemy"]:
            heavy = 1.26 if enemy.get("boss") and rnd % BOSS_WINDOW == 0 else 1.0
            if build.beat and rnd % BEAT_EVERY == 0:
                heavy *= BEAT_TAKEN
            if e_pred > 0:
                self._update("k_enemy", "n_enemy", dmg / (e_pred * heavy))


@dataclass
class FightEstimate:
    win: float       # P(enemy dies within the horizon)
    death: float     # P(player dies within the horizon)
    hp_after: float  # mean player HP+shield at the end, over surviving samples


RAINBOW_FLEE = {"prism_wind": 0.20}   # 风隙羽标: flee +20 pp after the usual cap; its other bonuses are in the stats


def rainbow_flee_bonus(state: dict) -> float:
    """Flee bonus of the unlocked rainbow items (their permanent skills apply from the first drop)."""
    return sum(RAINBOW_FLEE.get(r.get("id"), 0.0) for r in (state.get("atlas") or {}).get("rainbow") or []
               if r.get("unlocked"))


def run_build(state: dict, affix_build: Build) -> Build:
    """Combat Build for the current moment of a run: live stats + permanent affixes.
    The server's secondary stats are totals (gear, blessings, active field items); `mitigation`
    (打击减伤) only exists on servers since 2026-11-25."""
    run, sec = state["run"], state["player"].get("secondary") or {}
    es = run["effective_stats"]
    mult = float(run.get("damage_multiplier") or 1)
    reduction = float(sec["mitigation"]) if sec.get("mitigation") is not None else affix_build.reduction
    affix_build = replace(affix_build, **heirlooms.build_effects(state.get("heirlooms")),
                          depth_shred_cap=float((sec.get("condition") or {}).get("depth_cap") or 0.0),
                          flee_bonus=rainbow_flee_bonus(state))
    return replace(affix_build, hp=run["max_hp"], atk=es["atk"], defense=es["def"], agi=es["agi"], luk=es["luk"],
                   reduction=reduction,
                   pierce=sec.get("pierce") or affix_build.pierce, leech=sec.get("leech") or affix_build.leech,
                   combo=sec.get("combo") or affix_build.combo, boss_bonus=sec.get("boss_bonus") or affix_build.boss_bonus,
                   crit_mult=sec.get("crit_multiplier") or affix_build.crit_mult,
                   dealt_mult=mult, taken_mult=mult, atk_pct=0.0, def_pct=0.0,
                   last_stand=False)  # once per run and its state is not exposed: assume spent


def battle_enemy(state: dict, enemy: dict) -> dict:
    """The enemy as the damage formula sees it: defence after the blessings' deep-floor armour shred of
    the current floor (`player.secondary.condition.depth_shred`, servers since 2026-11-25)."""
    condition = (state["player"].get("secondary") or {}).get("condition") or {}
    shred = float(condition.get("depth_shred") or 0.0)
    return {**enemy, "def": enemy["def"] * (1 - shred)} if shred > 0 else enemy


def estimate(build: Build, state: dict, cal: Calibration, samples: int = 600,
             rng: np.random.Generator | None = None, rules: ServerRules = LIVE_RULES) -> FightEstimate:
    """Outcome of the fight segment that starts now: up to the next pause (boss window, elite pause)
    or the round cap. `win` is P(the enemy dies within it)."""
    run = state["run"]
    enemy = battle_enemy(state, run["event"]["enemy"])
    boss = bool(enemy.get("boss"))
    start, rounds = fight_segment(enemy, rules)
    kp, ke = cal.conservative()
    tuned = replace(build, dealt_mult=build.dealt_mult * kp, taken_mult=build.taken_mult * ke)
    n = samples
    hp, shield, ehp, _, _ = fight(tuned, np.full(n, float(run["hp"])), np.full(n, float(run.get("shield") or 0)),
                                  enemy, boss=boss, ehp=np.full(n, float(enemy["hp"])), start_round=start,
                                  rounds=rounds, rng=rng or np.random.default_rng())
    dead = hp <= 0
    alive = ~dead
    return FightEstimate(win=float((alive & (ehp <= 0)).mean()), death=float(dead.mean()),
                         hp_after=float((hp + shield)[alive].mean()) if alive.any() else 0.0)


def chain_risk(build: Build, state: dict, cal: Calibration, fights: int = 4, samples: int = 600,
               map_id: str = "mall", rng: np.random.Generator | None = None, elites: bool = False,
               rules: ServerRules = LIVE_RULES) -> float:
    """P(death) if the next `fights` floors each hold a normal monster, fought back to back with
    no healing in between (what a rush does). Used to decide between rush and single steps.
    elites: a rush with 自动打精英 on - each fight is an elite with the floor's elite share of the
    combats, fought up to its pause; an elite still standing there halts that sample's rush."""
    run = state["run"]
    rng = rng or np.random.default_rng()
    kp, ke = cal.conservative()
    tuned = replace(build, dealt_mult=build.dealt_mult * kp, taken_mult=build.taken_mult * ke)
    hp = np.full(samples, float(run["hp"]))
    shield = np.full(samples, float(run.get("shield") or 0))
    going = np.ones(samples, dtype=bool)   # still rushing (elites only: a pause halts the rush)
    for k in range(1, fights + 1):
        floor = run["floor"] + k
        new_hp, new_shield, *_ = fight(tuned, hp, shield, foe_at("normal", floor, map_id, tuned), boss=False, rng=rng)
        if elites:
            mix = floor_mix(floor, rules)
            elite = {**foe_at("elite", floor, map_id, tuned), "elite": True, "rounds_fought": 0}
            first, rounds = fight_segment(elite, rules)
            e_hp, e_shield, e_ehp, *_ = fight(tuned, hp, shield, elite, boss=False, start_round=first,
                                              rounds=rounds, rng=rng)
            met = going & (rng.random(samples) < mix["elite"] / (mix["elite"] + mix["normal"]))
            new_hp, new_shield = np.where(met, e_hp, new_hp), np.where(met, e_shield, new_shield)
        hp, shield = np.where(going, new_hp, hp), np.where(going, new_shield, shield)
        if elites:
            going &= ~(met & (e_ehp > 0))
        hp = np.where(hp <= 0, -1.0, hp)  # dead samples stay dead
    return float((hp <= 0).mean())
