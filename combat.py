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

from model import BOSS_WINDOW, Build, enemy, fight

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
        p_pred = raw_hit(build.eff_atk, enemy["def"] * (1 - min(0.75, build.pierce))) * build.dealt_mult * boss_mult
        e_pred = raw_hit(enemy["atk"], build.eff_def) * build.taken_mult * (1 - build.reduction)
        for rnd, dmg in hits["player"]:
            if rnd > 1 and p_pred > 0:  # round 1 carries one-off bonuses
                berserk = 1 + min(BERSERK_CAP, (rnd - 1) * build.berserk)
                self._update("k_player", "n_player", dmg / (p_pred * berserk))
        for rnd, dmg in hits["enemy"]:
            heavy = 1.26 if enemy.get("boss") and rnd % BOSS_WINDOW == 0 else 1.0
            if e_pred > 0:
                self._update("k_enemy", "n_enemy", dmg / (e_pred * heavy))


@dataclass
class FightEstimate:
    win: float       # P(enemy dies within the horizon)
    death: float     # P(player dies within the horizon)
    hp_after: float  # mean player HP+shield at the end, over surviving samples


def run_build(state: dict, affix_build: Build) -> Build:
    """Combat Build for the current moment of a run: live stats + permanent affixes."""
    run, sec = state["run"], state["player"].get("secondary") or {}
    es = run["effective_stats"]
    mult = float(run.get("damage_multiplier") or 1)
    return replace(affix_build, hp=run["max_hp"], atk=es["atk"], defense=es["def"], agi=es["agi"], luk=es["luk"],
                   pierce=sec.get("pierce") or affix_build.pierce, leech=sec.get("leech") or affix_build.leech,
                   combo=sec.get("combo") or affix_build.combo, boss_bonus=sec.get("boss_bonus") or affix_build.boss_bonus,
                   crit_mult=sec.get("crit_multiplier") or affix_build.crit_mult,
                   dealt_mult=mult, taken_mult=mult, atk_pct=0.0, def_pct=0.0,
                   last_stand=False)  # once per run and its state is not exposed: assume spent


def estimate(build: Build, state: dict, cal: Calibration, samples: int = 600,
             rng: np.random.Generator | None = None) -> FightEstimate:
    run = state["run"]
    enemy = run["event"]["enemy"]
    boss = bool(enemy.get("boss"))
    start = (enemy.get("rounds_fought") or 0) + 1
    rounds = BOSS_WINDOW - ((start - 1) % BOSS_WINDOW) if boss else 80
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
               map_id: str = "mall", rng: np.random.Generator | None = None) -> float:
    """P(death) if the next `fights` floors each hold a normal monster, fought back to back with
    no healing in between (what a rush does). Used to decide between rush and single steps."""
    run = state["run"]
    rng = rng or np.random.default_rng()
    kp, ke = cal.conservative()
    tuned = replace(build, dealt_mult=build.dealt_mult * kp, taken_mult=build.taken_mult * ke)
    hp = np.full(samples, float(run["hp"]))
    shield = np.full(samples, float(run.get("shield") or 0))
    for k in range(1, fights + 1):
        hp, shield, *_ = fight(tuned, hp, shield, enemy("normal", run["floor"] + k, map_id), boss=False, rng=rng)
        hp = np.where(hp <= 0, -1.0, hp)  # dead samples stay dead
    return float((hp <= 0).mean())
