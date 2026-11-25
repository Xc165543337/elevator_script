"""Field items (局内道具): server specs and their effect on a live fight.

The live state already folds part of the active field modifiers into what the agent reads
(`effective_stats` carries attack / defence / agility, `player.secondary` carries pierce / follow-up /
leech / boss bonus). Everything else - dodge, crit, damage reduction, first-hit bonus, thorns and the
guard / burst charges - only exists as item ids, so it is rebuilt here from the server specs
(data/field_items.json, exported from the game's content.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

from model import Build

_DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "field_items.json"),
                       encoding="utf-8"))
SPECS: dict[str, dict] = _DATA["items"]
MOD_CAPS: dict[str, tuple[float, float]] = {k: tuple(v) for k, v in _DATA["mod_caps"].items()}
TIMED_SLOTS = 3                      # server FIELD_TIMED_SLOTS
FIGHT_MODES = ("timed", "guard", "burst")


def _cap(key: str, value: float) -> float:
    lo, hi = MOD_CAPS.get(key, (-float("inf"), float("inf")))
    return min(hi, max(lo, value))


def active_mods(run: dict) -> dict[str, float]:
    """Uncapped modifier totals of worn field gear and active floor-limited buffs (server order)."""
    totals: dict[str, float] = {}
    for entry in [*(run.get("field_buffs") or []), *(run.get("field_slots") or [])]:
        spec = SPECS.get((entry or {}).get("id", ""))
        for key, value in ((spec or {}).get("mods") or {}).items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def worn_mods(run: dict) -> dict[str, float]:
    """Modifier totals of worn field gear only (they last the whole run)."""
    totals: dict[str, float] = {}
    for entry in run.get("field_slots") or []:
        for key, value in ((SPECS.get((entry or {}).get("id", "")) or {}).get("mods") or {}).items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def live_build(run: dict, build: Build, mitigation_in_stats: bool = False) -> Build:
    """Add the field effects that the state does not expose as numbers. `mitigation_in_stats`: the
    server's secondary stats carry 打击减伤 (newer servers), which already includes field items."""
    mods = active_mods(run)
    extra_reduction = 0.0 if mitigation_in_stats else _cap("mitigation", mods.get("mitigation", 0.0))
    out = replace(build, dodge_bonus=_cap("dodge", mods.get("dodge", 0.0)),
                  crit_bonus=build.crit_bonus + _cap("crit", mods.get("crit", 0.0)),
                  reduction=build.reduction + extra_reduction,
                  heavy=build.heavy + _cap("first", mods.get("first", 0.0)),
                  thorns=build.thorns + _cap("thorns", mods.get("thorns", 0.0)))
    for charge in run.get("field_charges") or []:
        spec = SPECS.get(charge.get("id", ""), {})
        left = int(charge.get("left") or 0)
        if spec.get("mode") == "guard":
            out = replace(out, guard_hits=left, guard_reduction=spec["reduction"])
        elif spec.get("mode") == "burst" and not spec.get("blood"):
            out = replace(out, burst_hits=left, burst_mult=spec["mult"], burst_sure=bool(spec.get("sure")))
    return out


@dataclass(frozen=True)
class FieldPrep:
    """A field item that changes the coming fight (not a heal)."""
    id: str
    name: str
    spec: dict

    @property
    def mode(self) -> str:
        return self.spec["mode"]


def fight_preparations(run: dict) -> list[FieldPrep]:
    """Usable timed / guard / burst items, under the server's activation rules."""
    buffs = {b.get("id") for b in run.get("field_buffs") or []}
    charged = {SPECS.get(c.get("id", ""), {}).get("mode") for c in run.get("field_charges") or []}
    if any(SPECS.get(c.get("id", ""), {}).get("blood") for c in run.get("field_charges") or []):
        return []                                        # a blood charge locks healing and shields
    out = []
    for item in run.get("field") or []:
        spec = SPECS.get(item.get("id", ""))
        if not spec or not item.get("quantity") or spec["mode"] not in FIGHT_MODES:
            continue
        if spec.get("blood") or spec.get("judgement") or spec.get("cost_hp"):
            continue                                     # all-in items stay manual
        if spec["mode"] in charged or item["id"] in buffs:
            continue
        if spec["mode"] == "timed" and len(buffs) >= TIMED_SLOTS:
            continue
        out.append(FieldPrep(item["id"], item.get("name") or spec["name"], spec))
    return out


def compatible(first: FieldPrep, second: FieldPrep, run: dict) -> bool:
    """Whether `second` can still be activated after `first` in the same encounter."""
    if first.id == second.id or (first.mode == second.mode and first.mode in ("guard", "burst")):
        return False
    if first.mode == second.mode == "timed":
        return len(run.get("field_buffs") or []) + 2 <= TIMED_SLOTS
    return True


def apply(run: dict, build: Build, prep: FieldPrep, mods: dict | None = None) -> tuple[Build, dict]:
    """Build after activating `prep`. `mods` = modifier totals before it (default: the live ones).
    Percentage modifiers act on base stats; each key is capped on the total, like the server."""
    mods = dict(active_mods(run) if mods is None else mods)
    spec = prep.spec
    if prep.mode == "guard":
        return replace(build, guard_hits=spec["hits"], guard_reduction=spec["reduction"]), mods
    if prep.mode == "burst":
        return replace(build, burst_hits=spec["hits"], burst_mult=spec["mult"],
                       burst_sure=bool(spec.get("sure"))), mods
    base = run.get("stats") or run["effective_stats"]
    delta = {}
    for key, value in (spec.get("mods") or {}).items():
        before = mods.get(key, 0.0)
        delta[key] = _cap(key, before + value) - _cap(key, before)
        mods[key] = before + value
    d = delta.get
    build = replace(
        build,
        atk=max(0.25 * base["atk"], build.atk + base["atk"] * d("atk", 0.0)),
        defense=max(0.0, build.defense + base["def"] * d("defense", 0.0)),
        agi=max(0.0, build.agi + d("agi", 0.0)),
        leech=min(0.42, build.leech + d("leech", 0.0)), combo=min(0.70, build.combo + d("combo", 0.0)),
        pierce=min(0.75, build.pierce + d("pierce", 0.0)), boss_bonus=min(1.2, build.boss_bonus + d("boss", 0.0)),
        reduction=build.reduction + d("mitigation", 0.0), crit_bonus=build.crit_bonus + d("crit", 0.0),
        dodge_bonus=build.dodge_bonus + d("dodge", 0.0), heavy=build.heavy + d("first", 0.0),
        thorns=min(0.7, build.thorns + d("thorns", 0.0)))
    return build, mods
