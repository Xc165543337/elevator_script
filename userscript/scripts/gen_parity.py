"""Generate tests/fixtures/parity.json from the Python reference implementation.

The JS port is checked against these frozen Python results in tests/parity.test.ts:
exact values for deterministic parts (monster growth, scoring, gear, calibration) and
statistical tolerances for the Monte Carlo parts (fight / chain-risk estimates, decisions).

Usage (from userscript/, needs numpy):  python scripts/gen_parity.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from combat import Calibration, chain_risk, estimate, parse_battle_log, run_build  # noqa: E402
from gear import GearModel  # noqa: E402
from model import MAP_MULT, Build, enemy, fight, is_boss_floor  # noqa: E402
from policy import BossFarmPolicy, PolicyConfig  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "parity.json")
TRACE = os.path.join(ROOT, "docs", "sim_trace_sample.jsonl")
N = 40_000

CAMEL = {"defense": "defense", "boss_bonus": "bossBonus", "crit_mult": "critMult", "crit_bonus": "critBonus",
         "last_stand": "lastStand", "start_shield": "startShield", "dealt_mult": "dealtMult",
         "taken_mult": "takenMult", "def_pct": "defPct", "atk_pct": "atkPct"}

AFFIXES = Build(hp=666, atk=150, defense=70, agi=77, luk=21, pierce=0.26, leech=0.06, heavy=0.38,
                berserk=0.075, vampire=0.10, crit_mult=1.9)
# Run fields the policy never reads; dropped to keep the fixture small.
UNUSED_RUN_KEYS = {"floor_result", "adventure", "radar_scan", "recovery", "stats", "signal", "expires_at"}
PLAYER = {"loadout": {}, "secondary": {"crit": 0.15, "crit_multiplier": 1.9, "pierce": 0.26, "leech": 0.06}}


def js_build(b: Build) -> dict:
    return {CAMEL.get(k, k): v for k, v in asdict(b).items()}


def enemies() -> list:
    out = []
    floors = list(range(1, 61)) + list(range(61, 2001, 7))
    for map_id in MAP_MULT:
        for f in floors:
            for kind in ("normal", "elite"):
                out.append({"kind": kind, "floor": f, "map": map_id, "foe": enemy(kind, f, map_id)})
            if is_boss_floor(f):
                out.append({"kind": "boss", "floor": f, "map": map_id, "foe": enemy("boss", f, map_id)})
        for f in range(30, 2001, 20):
            out.append({"kind": "boss", "floor": f, "map": map_id, "foe": enemy("boss", f, map_id)})
    return out


def fights() -> list:
    builds = {
        "affixes": AFFIXES,
        "combo_thorns": replace(AFFIXES, combo=0.35, thorns=0.25, boss_bonus=0.4, start_shield=25),
        "desperate": replace(AFFIXES, desperate=True, last_stand=True, crit_bonus=0.13, reduction=-0.1),
        "balance": AFFIXES.with_temp_gear("balance", "apron"),
        "slow": replace(AFFIXES, agi=5, leech=0.0, vampire=0.0, heavy=0.0),
    }
    cases = []
    for name, b in builds.items():
        for kind, floor, hp, shield, start in (("normal", 300, 500, 0, 1), ("elite", 600, 400, 25, 1),
                                               ("elite", 900, 666, 0, 1), ("boss", 610, 666, 0, 1),
                                               ("boss", 1030, 350, 10, 5), ("boss", 1150, 666, 0, 3)):
            foe = enemy(kind, floor)
            boss = kind == "boss"
            rounds = 4 - ((start - 1) % 4) if boss else 80
            rng = np.random.default_rng(len(cases))
            out_hp, out_sh, out_ehp, _, _ = fight(b, np.full(N, float(hp)), np.full(N, float(shield)), foe, boss=boss,
                                                  start_round=start, rounds=rounds, rng=rng)
            alive = out_hp > 0
            cases.append({"build": js_build(b), "name": name, "foe": foe, "hp": hp, "shield": shield, "boss": boss,
                          "startRound": start, "rounds": rounds, "samples": N,
                          "death": float((~alive).mean()), "win": float((alive & (out_ehp <= 0)).mean()),
                          "hpAfter": float(out_hp[alive].mean()) if alive.any() else 0.0})
    return cases


def chains() -> list:
    cal = Calibration(path=os.devnull)
    out = []
    for floor, hp in ((100, 666), (600, 400), (1155, 300), (1155, 666), (900, 500)):
        state = {"run": {"floor": floor, "hp": hp, "max_hp": 666, "shield": 0,
                         "effective_stats": {"hp": 666, "atk": 150, "def": 70, "agi": 77, "luk": 21}},
                 "player": PLAYER}
        b = run_build(state, AFFIXES)
        out.append({"state": state, "risk": chain_risk(b, state, cal, samples=N, rng=np.random.default_rng(floor))})
    return out


def trace_states() -> list:
    with open(TRACE, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    return [line for line in lines if line.get("run")]  # drop the state after the run ended


def decisions(lines: list) -> tuple[list, list]:
    """(states, decisions). Each decision applies its `patch` to run of states[index]."""
    cal = Calibration(path=os.devnull)
    cfg = PolicyConfig(samples=4000)
    states, out = [], []
    for i, line in enumerate(lines):
        run = {k: v for k, v in line["run"].items() if k not in UNUSED_RUN_KEYS}
        run["bag"] = [{k: v for k, v in it.items() if k != "actions"} for it in run.get("bag") or []]
        states.append({"run": run, "controls": line["controls"]})
        # Second variant without a safe box: skips loot securing so fight / rush / event logic is exercised too.
        for variant, patch in (("trace", {}), ("no_safe_box", {"safe": [], "safe_capacity": 0})):
            state = {"run": {**run, **patch}, "controls": line["controls"], "player": PLAYER, "atlas": {"items": []}}
            policy = BossFarmPolicy(cal, cfg)
            policy._affix_build, policy._loadout = AFFIXES, ()
            d = policy.decide(state)
            entry = {"index": i, "variant": variant, "patch": patch, "command": d.command, "params": d.params}
            build = run_build(state, AFFIXES)
            foe = (run.get("event") or {}).get("enemy")
            if foe and any(c["command"] == "fight" for c in state["controls"]):
                entry["death"] = estimate(build, state, cal, samples=N, rng=np.random.default_rng(i)).death
            elif any(c["command"] == "rush" for c in state["controls"]):
                entry["chainRisk"] = chain_risk(build, state, cal, samples=N, map_id=run.get("map") or "mall",
                                                rng=np.random.default_rng(i))
            out.append(entry)
    return states, out


def scores(lines: list) -> dict:
    events, wears = {}, {}
    for line in lines:
        run = line["run"]
        hp_frac = run["hp"] / run["max_hp"]
        for c in line["controls"]:
            if c["command"] == "event" and c.get("description"):
                for capped in (False, True):
                    key = f"{c['description']}|{hp_frac:.6f}|{capped}"
                    events[key] = {"desc": c["description"], "hpFrac": hp_frac, "maxHp": run["max_hp"], "capped": capped,
                                   "score": BossFarmPolicy._score_option(c["description"], hp_frac, run["max_hp"], capped)}
        texts = [(run.get("field_offer") or {}).get("description")]
        texts += [s.get("description") for s in run.get("field_slots") or [] if isinstance(s, dict)]
        texts += [it.get("description") for it in run.get("field") or []]
        for t in filter(None, texts):
            wears[t] = BossFarmPolicy._wear_score(t)
    extra = ["敏捷检定94%成功→得1件紫色底起跳物资；失败(6%)扣约133HP", "100%→得十倍浓缩冰红茶",
             "耗 20% 当前血(约101)，100%→本局攻+20%", "无消耗，离开", "70%成功→回30%血、防+10%；失败扣约80HP",
             "耗约50金币，偷看下一层"]
    for desc in extra:
        for hp_frac in (1.0, 0.4):
            events[f"{desc}|{hp_frac}"] = {"desc": desc, "hpFrac": hp_frac, "maxHp": 666, "capped": False,
                                           "score": BossFarmPolicy._score_option(desc, hp_frac, 666, False)}
    for desc in ("本局佩戴，占 2 个临装位之一 · 闪避率+11% · 攻击+13%", "本局佩戴 · 减伤+17% · 攻击-8%",
                 "本局佩戴 · 对首领伤害+20% · 逃跑率+10% · 首回合攻击+30%"):
        wears[desc] = BossFarmPolicy._wear_score(desc)
    return {"events": list(events.values()), "wears": [{"desc": k, "score": v} for k, v in wears.items()]}


def calibration(lines: list) -> dict:
    cal = Calibration(path=os.devnull)
    steps = []
    for line in lines:
        foe = (line["run"].get("event") or {}).get("enemy")
        battle = (line.get("result") or {}).get("battle") or {}
        if line["pick"][0] == "fight" and foe and battle.get("log"):
            state = {"run": line["run"], "player": PLAYER}
            b = run_build(state, AFFIXES)
            cal.observe(b, foe, battle["log"])
            steps.append({"build": js_build(b), "enemy": foe, "log": battle["log"], "hits": parse_battle_log(battle["log"])})
    return {"steps": steps, "final": {"k_player": cal.k_player, "k_enemy": cal.k_enemy,
                                      "n_player": cal.n_player, "n_enemy": cal.n_enemy}}


def gear() -> dict:
    """Synthetic warehouse exercising stats, 2/4-piece sets, stacked affixes and flags."""
    def item(eid, slot, set_name, stats, affixes, desc=""):
        return {"equip_id": eid, "name": f"item{eid}", "slot": slot, "rarity": "gold", "set_name": set_name,
                "stats": stats, "affixes": [{"id": a, "name": a} for a in affixes], "description": desc}
    set_desc = "2件 · 攻击+12 · 命中吸血+6%\n4件 · 对首领伤害+25% · 越战越勇+3%"
    equipment = [
        item(1, "武器", "猎手", {"atk": 40, "luk": 3}, ["pierce", "heavy_strike"], set_desc),
        item(2, "护甲", "猎手", {"hp": 120, "def": 30}, ["pierce", "vampire"], set_desc),
        item(3, "挂件", "猎手", {"agi": 9}, ["high_roller", "wager"], set_desc),
        item(4, "核心", "猎手", {"hp": 60, "atk": 8}, ["berserk", "last_stand", "greed", "treasure"], set_desc),
        item(5, "武器", None, {"atk": 55}, ["thorns", "shield_start"]),
        item(6, "护甲", "铁壁", {"def": 45}, [], "2件 · 防御+20 · 反甲刺猬+10%\n4件 · 出门套盾+40"),
    ]
    player = {"loadout": {"武器": 1, "护甲": 2, "挂件": 3, "核心": 4},
              "stat_sources": {
                  "hp": [{"source": "基础", "op": "base", "value": 400}, {"source": "#2 item2", "op": "add", "value": 120},
                         {"source": "天赋", "op": "add", "value": 30}],
                  "atk": [{"source": "基础", "op": "base", "value": 70}, {"source": "猎手 · 2件", "op": "add", "value": 12},
                          {"source": "训练", "op": "mul", "value": 5}],
                  "def": [{"source": "基础", "op": "base", "value": 25}],
                  "agi": [{"source": "基础", "op": "base", "value": 40}],
                  "luk": [{"source": "基础", "op": "base", "value": 10}]}}
    state = {"equipment": equipment, "player": player}
    gm = GearModel(state)
    by_id = {i.equip_id: i for i in gm.items}
    builds = {"current": js_build(gm.build(gm.current)),
              "mixed": js_build(gm.build(tuple(by_id[i] for i in (5, 6, 3, 4))))}
    return {"state": state, "base": gm.base, "sets": {k: {str(p): v for p, v in t.items()} for k, t in gm.sets.items()},
            "builds": builds}


def main() -> None:
    lines = trace_states()
    states, decided = decisions(lines)
    data = {"affixes": js_build(AFFIXES), "player": PLAYER, "enemies": enemies(), "fights": fights(),
            "chains": chains(), "states": states, "decisions": decided, "scores": scores(lines),
            "calibration": calibration(lines), "gear": gear()}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"{OUT}: {os.path.getsize(OUT) / 1000:.0f} kB, {len(data['decisions'])} decisions, "
          f"{len(data['fights'])} fights, {len(data['enemies'])} enemies")


if __name__ == "__main__":
    main()
