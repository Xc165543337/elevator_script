"""Gear -> Build conversion and build search.

Verified in game: identical affixes on different items do NOT stack (3x 破甲 still shows 26%
pierce). Each affix counts once; set bonuses are added on top of it.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from model import Build

SLOTS = ("武器", "护甲", "挂件", "核心")
STAT_KEYS = {"生命": "hp", "攻击": "atk", "防御": "def", "敏捷": "agi", "幸运": "luk"}
# Affix id -> (Build field, value per copy)
AFFIX_EFFECT = {
    "pierce": ("pierce", 0.26), "vampire": ("vampire", 0.10), "heavy_strike": ("heavy", 0.38),
    "berserk": ("berserk", 0.075), "thorns": ("thorns", 0.25), "shield_start": ("start_shield", 25),
}
FLAG_AFFIXES = {"wager": "desperate", "last_stand": "last_stand"}
# Set-bonus phrases that map onto Build fields (fractions for %-values).
SET_SPECIAL = {"命中吸血": "leech", "追击率": "combo", "越战越勇": "berserk", "对首领伤害": "boss_bonus",
               "反甲刺猬": "thorns", "出门套盾": "start_shield"}
# Affixes with no effect on boss farming in the Mall.
DEAD_AFFIXES = {"treasure", "loud", "kindhearted", "vulture", "rescue", "sneak", "fold", "scavenge",
                "slippery", "veteran", "greed"}


@dataclass(frozen=True)
class Item:
    equip_id: int
    name: str
    slot: str
    rarity: str
    set_name: str | None
    stats: tuple            # ((stat, value), ...)
    affixes: tuple          # affix ids
    affix_names: tuple

    @classmethod
    def from_state(cls, e: dict) -> "Item":
        return cls(e["equip_id"], e["name"], e["slot"], e["rarity"], e.get("set_name"),
                   tuple(sorted(e["stats"].items())), tuple(a["id"] for a in e["affixes"]),
                   tuple(a["name"] for a in e["affixes"]))

    def short(self) -> str:
        return f"#{self.equip_id} {self.name} [{self.set_name or '-'}] {dict(self.stats)} {'/'.join(self.affix_names)}"


def parse_set_bonuses(equipment: list[dict]) -> dict[str, dict[int, dict]]:
    """{set_name: {2: {...}, 4: {...}}} parsed from item descriptions."""
    sets: dict[str, dict[int, dict]] = {}
    for e in equipment:
        name = e.get("set_name")
        if not name or name in sets:
            continue
        tiers = {}
        for pieces, body in re.findall(r"(\d)件 · ([^\n]+)", e.get("description", "")):
            bonus = {}
            for key, value, pct in re.findall(r"([一-龥]+)\+(\d+(?:\.\d+)?)(%?)", body):
                field = STAT_KEYS.get(key) or SET_SPECIAL.get(key)
                if field:
                    bonus[field] = bonus.get(field, 0) + (float(value) / 100 if pct else float(value))
            tiers[int(pieces)] = bonus
        sets[name] = tiers
    return sets


def base_stats(player: dict, gear_sets: set[str]) -> dict[str, float]:
    """Permanent stats without any equipped item or gear-set bonus."""
    base = {}
    for stat, sources in player["stat_sources"].items():
        total = 0.0
        for src in sources:
            label = src["source"]
            if "#" in label or any(label.startswith(s + " ·") for s in gear_sets):
                continue
            if src["op"] in ("base", "add"):
                total += src["value"]
        base[stat] = total
    return base


class GearModel:
    def __init__(self, state: dict):
        self.items = [Item.from_state(e) for e in state["equipment"]]
        self.sets = parse_set_bonuses(state["equipment"])
        self.base = base_stats(state["player"], set(self.sets))
        self.by_slot = {s: [i for i in self.items if i.slot == s] for s in SLOTS}
        loadout = state["player"]["loadout"]
        by_id = {i.equip_id: i for i in self.items}
        self.current = tuple(by_id[loadout[s]] for s in SLOTS if loadout.get(s) in by_id)  # skip empty / unknown slots

    def build(self, items: tuple[Item, ...], label: str = "") -> Build:
        stats = dict(self.base)
        extra: Counter = Counter()
        for it in items:
            for k, v in it.stats:
                stats[k] = stats.get(k, 0) + v
        for set_name, count in Counter(i.set_name for i in items if i.set_name).items():
            for pieces, bonus in self.sets.get(set_name, {}).items():
                if count >= pieces:
                    for k, v in bonus.items():
                        if k in stats:
                            stats[k] += v
                        else:
                            extra[k] += v
        flags = {}
        crit_mult = 1.6
        for a in {a for it in items for a in it.affixes}:  # each affix counts once
            if a in AFFIX_EFFECT:
                field, value = AFFIX_EFFECT[a]
                extra[field] += value
            elif a in FLAG_AFFIXES:
                flags[FLAG_AFFIXES[a]] = True
            elif a == "high_roller":
                crit_mult = 1.9
            elif a == "greed":  # 贪婪诅咒: +20% damage taken
                extra["greed"] += 0.20
        return Build(hp=stats["hp"], atk=stats["atk"], defense=stats["def"], agi=stats["agi"], luk=stats["luk"],
                     pierce=extra["pierce"], leech=extra["leech"], combo=extra["combo"], heavy=extra["heavy"],
                     berserk=extra["berserk"], vampire=extra["vampire"], thorns=extra["thorns"],
                     boss_bonus=extra["boss_bonus"], start_shield=extra["start_shield"], crit_mult=crit_mult,
                     taken_mult=1.0 + extra["greed"],
                     label=label, **flags)
