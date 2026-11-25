"""Search the warehouse for the gear set that maximises bosses per run (offline, no game calls
other than reading the state once).

    python optimize.py                 # search and print the top builds
    python optimize.py --runs 150      # more samples per evaluation (slower, less noise)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

from gear import DEAD_AFFIXES, SLOTS, GearModel, Item
from model import simulate_runs

TEMP_GEAR = ("apron", "apron")  # best Mall temp gear per model (see README)
ADRENALINE = 2                    # shots used on F1
_GM: GearModel | None = None
_ARGS: dict = {}


def _init(state: dict, args: dict) -> None:
    global _GM, _ARGS
    _GM, _ARGS = GearModel(state), args


def _score(combo_ids: tuple[int, ...]) -> tuple[tuple[int, ...], float, float, float]:
    by_id = {i.equip_id: i for i in _GM.items}
    build = _GM.build(tuple(by_id[i] for i in combo_ids)).with_temp_gear(*_ARGS["temp"])
    r = simulate_runs(build, runs=_ARGS["runs"], heals=_ARGS["heals"], max_floor=_ARGS["max_floor"],
                      adrenaline=ADRENALINE, seed=7)
    return combo_ids, r.bosses, r.floor, r.boss_windows_per_boss


def useful_candidates(gm: GearModel, slot: str, keep: int) -> list[Item]:
    """Red/gold items with at least one useful affix, ranked by a cheap static score."""
    def static(i: Item) -> float:
        s = dict(i.stats)
        useful = sum(a not in DEAD_AFFIXES for a in i.affixes)
        return useful * 30 + s.get("atk", 0) * 1.5 + s.get("hp", 0) * 0.25 + s.get("def", 0) + s.get("agi", 0) * 1.2
    pool = [i for i in gm.by_slot[slot] if i.rarity in ("red", "gold")]
    return sorted(pool, key=static, reverse=True)[:keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(os.path.dirname(__file__), "logs", "state_snapshot.json"))
    ap.add_argument("--runs", type=int, default=80)
    ap.add_argument("--heals", type=int, default=30)
    ap.add_argument("--max-floor", type=int, default=1800)
    ap.add_argument("--keep", type=int, default=12, help="candidates per slot")
    ap.add_argument("--rounds", type=int, default=3, help="coordinate-descent passes")
    args = ap.parse_args()
    with open(args.state, encoding="utf-8") as fh:
        state = json.load(fh)
    gm = GearModel(state)
    opts = {"runs": args.runs, "heals": args.heals, "max_floor": args.max_floor, "temp": TEMP_GEAR}
    cands = {s: useful_candidates(gm, s, args.keep) for s in SLOTS}
    for s in SLOTS:  # always allow the currently worn item
        if gm.current[SLOTS.index(s)] not in cands[s]:
            cands[s].append(gm.current[SLOTS.index(s)])
    results: dict[tuple, tuple] = {}

    with ProcessPoolExecutor(initializer=_init, initargs=(state, opts)) as pool:
        def evaluate(combos):
            todo = [c for c in combos if c not in results]
            for ids, bosses, floor, wpb in pool.map(_score, todo):
                results[ids] = (bosses, floor, wpb)

        current = tuple(i.equip_id for i in gm.current)
        evaluate([current])
        # 1) coordinate descent from the current loadout
        best = current
        for _ in range(args.rounds):
            improved = False
            for k, slot in enumerate(SLOTS):
                combos = [best[:k] + (it.equip_id,) + best[k + 1:] for it in cands[slot]]
                evaluate(combos)
                top = max(combos + [best], key=lambda c: results[c][0])
                if top != best:
                    best, improved = top, True
            if not improved:
                break
        # 2) full 4-piece sets: best item of that set per slot (by static score), plus 2+2 mixes
        set_names = {i.set_name for i in gm.items if i.set_name}
        set_combos = []
        for name in set_names:
            per_slot = [[i for i in useful_candidates(gm, s, 60) if i.set_name == name][:2] for s in SLOTS]
            if all(per_slot):
                set_combos += [tuple(i.equip_id for i in c) for c in itertools.product(*per_slot)]
        evaluate(set_combos)
        # 3) polish the best few with another descent pass
        for start in sorted(results, key=lambda c: results[c][0], reverse=True)[:3]:
            local = start
            for k, slot in enumerate(SLOTS):
                combos = [local[:k] + (it.equip_id,) + local[k + 1:] for it in cands[slot]]
                evaluate(combos)
                local = max(combos + [local], key=lambda c: results[c][0])

    by_id = {i.equip_id: i for i in gm.items}
    ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"evaluated {len(results)} builds; current = {results[current][0]:.1f} bosses, F{results[current][1]:.0f}")
    for ids, (bosses, floor, wpb) in ranked[:8]:
        b = gm.build(tuple(by_id[i] for i in ids))
        print(f"\n{bosses:5.1f} bosses  F{floor:5.0f}  {wpb:.1f} windows/boss | HP {b.hp:.0f} ATK {b.atk:.0f} DEF {b.defense:.0f} "
              f"AGI {b.agi:.0f} LUK {b.luk:.0f} pierce {b.pierce:.0%} vamp {b.vampire:.0%} heavy {b.heavy:.0%} "
              f"berserk {b.berserk:.1%} leech {b.leech:.0%} boss+{b.boss_bonus:.0%} thorns {b.thorns:.0%} combo {b.combo:.0%}")
        for i in ids:
            print("   ", by_id[i].short())
    with open(os.path.join(os.path.dirname(__file__), "logs", "optimize_results.json"), "w", encoding="utf-8") as fh:
        json.dump([{"ids": list(k), "bosses": v[0], "floor": v[1], "windows_per_boss": v[2]} for k, v in ranked], fh)


if __name__ == "__main__":
    main()
