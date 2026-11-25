"""闻岑 heirloom blessings (闪耀祝福), live since 2026-11-25.

Every formal boss kill rolls once for a shining drop (3% at F8, rising with depth to 10%). A drop
comes from one of the two sets chosen for the run (追寻, equal odds), and inside the set pieces still
missing are more likely. A real run cannot start without that choice; practice runs need none.
Pieces are worn in four independent blessing slots; two and four pieces of a set give its bonuses.

State shape (`state["heirlooms"]`, read from the live web client):
    sets    {set id: {name, short, lore, two, four}}     in server order
    pieces  {piece id: {name, set, stats: [stat, value], lore}}
    owned   {piece id: count}
    slots   [piece id | None] * 4
    focus   [set id, set id] - the last choice;  needs_focus - the running run still needs one
"""
from __future__ import annotations

import re
from collections import Counter


class FocusError(ValueError):
    """No valid pair of blessing sets to pursue."""


def resolve_focus(heirlooms: dict | None, wanted: str | None) -> list[str] | None:
    """The two set ids to pursue, or None on a server without blessings.
    `wanted`: two sets by id or by number in server order (1-based, like the chat command 追寻 1 8),
    separated by commas or spaces; None reuses the last choice."""
    if not heirlooms:
        return None
    sets = list(heirlooms.get("sets") or {})
    if wanted:
        picks = []
        for token in filter(None, re.split(r"[,\s]+", wanted.strip())):
            if token in sets:
                picks.append(token)
            elif token.isdigit() and 1 <= int(token) <= len(sets):
                picks.append(sets[int(token) - 1])
            else:
                raise FocusError(f"unknown blessing set {token!r}; choose from {describe_sets(heirlooms)}")
    else:
        picks = list(heirlooms.get("focus") or [])
    if len(picks) != 2 or picks[0] == picks[1]:
        raise FocusError("real runs need two different blessing sets to pursue: pass --focus, e.g. --focus 1,8 "
                         f"({describe_sets(heirlooms)})")
    return picks


def describe_sets(heirlooms: dict) -> str:
    return ", ".join(f"{i} = {s.get('name', key)}" for i, (key, s) in enumerate((heirlooms.get("sets") or {}).items(), 1))


def owned(heirlooms: dict | None) -> Counter:
    return Counter({k: int(v) for k, v in ((heirlooms or {}).get("owned") or {}).items() if v})


def new_pieces(before: Counter, after: Counter, heirlooms: dict | None) -> list[str]:
    """Names of the pieces gained between two ownership snapshots (one entry per copy)."""
    pieces = (heirlooms or {}).get("pieces") or {}
    gained = after - before
    return [pieces.get(key, {}).get("name", key) for key in sorted(gained) for _ in range(gained[key])]


def collection_lines(heirlooms: dict | None) -> list[str]:
    """One status line per set: pieces collected and worn."""
    if not heirlooms:
        return []
    have, worn = owned(heirlooms), set(filter(None, heirlooms.get("slots") or []))
    pieces = heirlooms.get("pieces") or {}
    focus = set(heirlooms.get("focus") or [])
    lines = []
    for i, (key, suit) in enumerate((heirlooms.get("sets") or {}).items(), 1):
        parts = [p for p, spec in pieces.items() if spec.get("set") == key]
        lines.append(f"{i}. {suit.get('name', key)}: {sum(1 for p in parts if have[p])}/{len(parts)} collected, "
                     f"{sum(1 for p in parts if p in worn)} worn{' (pursued)' if key in focus else ''}")
    return lines


def worn_sets(heirlooms: dict | None) -> Counter:
    """Pieces worn per set id."""
    pieces = (heirlooms or {}).get("pieces") or {}
    return Counter(pieces.get(key, {}).get("set") or key.split("-")[0]
                   for key in (heirlooms or {}).get("slots") or [] if key)


def build_effects(heirlooms: dict | None) -> dict:
    """Build fields set by worn set bonuses that the secondary stats do not carry. The stat bonuses
    themselves (pierce, crit, leech, ...) already arrive through `player.secondary`; set effects not
    listed here are not modelled yet."""
    worn = worn_sets(heirlooms)
    effects: dict = {}
    if worn["B06"] >= 2:
        effects["pierce_cap"] = 0.87      # 缺刻值夜表: pierce +12 pp above the 75% cap
    if worn["B06"] >= 4:
        effects["beat"] = True            # 抢拍: rounds 4, 8, ... act first, take x0.65
    return effects
