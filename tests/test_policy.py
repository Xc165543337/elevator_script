"""Unit tests for the pure parts of the agent (no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from combat import Calibration, parse_battle_log  # noqa: E402
from model import Build, enemy, fight  # noqa: E402
from policy import BossFarmPolicy, PolicyConfig  # noqa: E402

CAL = Calibration(k_player=1.0, k_enemy=1.0, path=os.devnull)
AFFIXES = Build(hp=666, atk=150, defense=70, agi=77, luk=21, pierce=0.26, leech=0.06, heavy=0.38,
                berserk=0.075, vampire=0.10, crit_mult=1.9)


def make_policy(**cfg) -> BossFarmPolicy:
    policy = BossFarmPolicy(CAL, PolicyConfig(**cfg))
    policy._affix_build, policy._loadout = AFFIXES, ()
    return policy


def make_state(hp=600, enemy=None, bag=None, field=None, safe=None, missing=(), controls=("fight", "extract")):
    run = {
        "floor": 100, "hp": hp, "max_hp": 666, "shield": 0,
        "effective_stats": {"hp": 666, "atk": 150, "def": 70, "agi": 77, "luk": 21},
        "event": {"kind": "combat", "enemy": enemy} if enemy else {"kind": "ready"},
        "bag": bag or [], "field": field or [], "safe": safe or [], "safe_capacity": 4,
        "field_slots": [None, None], "field_offer": None,
    }
    return {
        "run": run,
        "player": {"loadout": {}, "secondary": {"crit": 0.15, "crit_multiplier": 1.9, "pierce": 0.26, "leech": 0.06}},
        "controls": [{"command": c, "params": {}, "label": c} for c in controls],
        "atlas": {"items": [{"id": m, "unlocked": False, "rarity": "gold"} for m in missing]},
    }


def test_parse_battle_log():
    log = ["R1：你打 179（怪剩0）", "R2：你暴击打 296（怪剩361） ｜ 吸血 +17 ｜ 怪打 49（你剩489，盾挡3）",
           "R3：你出手被怪闪开 ｜ 怪出手被你闪开"]
    hits = parse_battle_log(log)
    assert hits["player"] == [(1, 179.0)]          # crits are excluded on purpose
    assert hits["enemy"] == [(2, 52.0)]            # shield-absorbed damage is included


def test_fight_engine_weak_enemy_is_safe_and_strong_enemy_is_deadly():
    rng = np.random.default_rng(1)
    hp, shield = np.full(500, 600.0), np.zeros(500)
    weak = {"hp": 50, "atk": 10, "def": 2, "agi": 5}
    strong = {"hp": 5000, "atk": 600, "def": 40, "agi": 30}
    assert (fight(AFFIXES, hp, shield, weak, boss=False, rng=rng)[0] <= 0).mean() == 0
    assert (fight(AFFIXES, hp, shield, strong, boss=False, rng=rng)[0] <= 0).mean() > 0.9


def test_monster_growth_matches_live_boss():
    assert enemy("boss", 1030)["hp"] == 4067      # observed in a live practice run
    assert enemy("normal", 2) == {"hp": 45, "atk": 11, "def": 2, "agi": 7}


def test_balance_gear_quadruples_damage_taken():
    rng = np.random.default_rng(2)
    foe = {"hp": 10**9, "atk": 100, "def": 0, "agi": 0}
    base = AFFIXES.__class__(**{**AFFIXES.__dict__, "agi": 0, "leech": 0, "vampire": 0})
    one = fight(base, np.full(400, 1e6), np.zeros(400), foe, boss=False, rounds=1, rng=rng)[0]
    four = fight(base.with_temp_gear("balance", "balance"), np.full(400, 1e6), np.zeros(400), foe, boss=False, rounds=1, rng=rng)[0]
    ratio = (1e6 - four[four < 1e6]).mean() / (1e6 - one[one < 1e6]).mean()
    assert 3.8 < ratio < 4.2


def test_uses_adrenaline_on_first_floor():
    bag = [{"id": "adrenaline", "index": 1, "name": "暴走肾上腺素"}]
    state = make_state(hp=666, bag=bag, controls=("rush",))
    state["run"]["floor"] = 1
    policy = make_policy()
    assert policy.decide(state).command == "use"
    assert policy.decide(state).command == "use"     # second stack
    assert policy.decide(state).command == "rush"    # cap of 2 reached


def test_fights_easy_enemy():
    state = make_state(enemy={"name": "mob", "hp": 50, "atk": 10, "def": 2, "agi": 5})
    assert make_policy().decide(state).command == "fight"


def test_heals_before_dangerous_fight():
    enemy = {"name": "elite", "elite": True, "hp": 1500, "atk": 260, "def": 45, "agi": 24}
    bag = [{"id": "field_surgery", "index": 3, "name": "战地手术匣"}]
    decision = make_policy().decide(make_state(hp=200, enemy=enemy, bag=bag))
    assert decision.command == "use" and decision.params == {"index": 3}


def test_extracts_when_hopeless_and_out_of_resources():
    enemy = {"name": "elite", "elite": True, "hp": 5000, "atk": 600, "def": 40, "agi": 30}
    decision = make_policy().decide(make_state(hp=200, enemy=enemy))
    assert decision.command == "extract"


def test_secures_missing_collectible():
    bag = [{"id": "tide_zero_rule", "index": 7, "name": "零潮水位原尺", "can_protect": True}]
    decision = make_policy().decide(make_state(bag=bag, missing=["tide_zero_rule"], controls=("rush",)))
    assert decision.command == "save_box" and decision.params == {"index": 7}


def test_event_scoring_prefers_safe_value():
    score = BossFarmPolicy._score_option
    assert score("敏捷检定94%成功→得1件紫色底起跳物资；失败(6%)扣约133HP", 1.0, 666) < 0
    assert score("100%→得十倍浓缩冰红茶", 1.0, 666) > 0
    assert score("耗 20% 当前血(约101)，100%→本局攻+20%", 0.9, 666) > 0


def test_wearable_scoring():
    assert BossFarmPolicy._wear_score("本局佩戴，占 2 个临装位之一 · 闪避率+11% · 攻击+13%") > 20
    assert BossFarmPolicy._wear_score("本局佩戴 · 减伤+17% · 攻击-8%") > 0


def test_rejected_move_is_not_repeated():
    enemy = {"name": "elite", "elite": True, "hp": 1500, "atk": 260, "def": 45, "agi": 24}
    field = [{"id": "field_caster", "name": "buff", "description": "持续 3 层 · 攻击+24%", "quantity": 1}]
    policy = make_policy()
    state = make_state(hp=150, enemy=enemy, field=field)
    first = policy.decide(state)
    assert first.command == "field_use"
    policy.ban(state["run"]["floor"], first)
    assert policy.decide(state).command != "field_use"


def test_affixes_do_not_stack():
    import json
    from gear import GearModel
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "state_snapshot.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        gm = GearModel(json.load(fh))
    by_id = {i.equip_id: i for i in gm.items}
    build = gm.build(tuple(by_id[i] for i in (12630, 1403, 12576, 2211)))  # three 破甲 items
    assert build.pierce == 0.26                      # verified in game
    assert (build.hp, build.atk, build.defense, build.agi) == (594, 123, 138, 61)


KIT = [{"id": "field_surgery", "index": 4, "name": "战地手术匣"}]


def test_deep_low_hp_steps_instead_of_rushing():
    state = make_state(hp=300, bag=KIT, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 1155
    decision = make_policy().decide(state)
    assert decision.command == "down", decision.reason


def test_shallow_full_hp_rushes():
    state = make_state(hp=666, bag=KIT, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 40
    assert make_policy().decide(state).command == "rush"


def test_secures_most_valuable_loot_and_keeps_supplies():
    bag = [{"id": "coffee", "index": 1, "name": "冰美式", "value": 300, "can_protect": True},
           {"id": "r08", "index": 2, "name": "red", "value": 35000, "can_protect": True},
           {"id": "g16", "index": 3, "name": "gold", "value": 4000, "can_protect": True}]
    d = make_policy().decide(make_state(bag=bag, controls=("rush",)))
    assert d.command == "save_box" and d.params == {"index": 2}


def test_swaps_only_when_clearly_better():
    safe = [{"id": f"x{i}", "index": i, "name": "x", "value": 10000} for i in range(1, 5)]
    better = [{"id": "r01", "index": 9, "name": "red", "value": 40000, "can_protect": True}]
    marginal = [{"id": "g01", "index": 9, "name": "gold", "value": 11000, "can_protect": True}]
    d = make_policy().decide(make_state(bag=better, safe=safe, controls=("rush",)))
    assert d.command == "swap_box" and d.params["index"] == 9
    assert make_policy().decide(make_state(bag=marginal, safe=safe, controls=("rush",))).command == "rush"


def test_select_loot_to_sell_rule():
    from farm import select_loot_to_sell
    stock = [
        {"stock_id": 1, "id": "g01", "kind": "collection", "rarity": "gold", "value": 5000, "quantity": 1},  # old
        {"stock_id": 2, "id": "g01", "kind": "collection", "rarity": "gold", "value": 7000, "quantity": 2},  # best
        {"stock_id": 3, "id": "r09", "kind": "collection", "rarity": "red", "value": 40000},                # only copy
        {"stock_id": 4, "id": "mat", "kind": "material", "rarity": "white", "value": 100, "quantity": 3},
        {"stock_id": 5, "id": "key", "kind": "material", "rarity": "blue", "value": 900},                   # blue kept
        {"stock_id": 6, "id": "coffee", "kind": "supply", "rarity": "green", "value": 300},               # supply kept
        {"stock_id": 7, "id": "crate", "kind": "crate", "rarity": "gold", "value": 9000},                 # crate kept
        {"stock_id": 8, "id": "old", "kind": "material", "rarity": "purple", "value": 50},                  # pre-run
        {"stock_id": 9, "id": "pie", "kind": "material", "rarity": "purple", "value": 800},                 # new
    ]
    picked = {(it["stock_id"], q) for it, q in select_loot_to_sell(stock, {1, 8}, {"red", "gold"},
                                                                     {"purple", "green", "white"})}
    # g01: keep one unit of the 7000 entry -> sell its 2nd unit and the 5000 copy (old copies included)
    assert picked == {(2, 1), (1, 1), (4, 3), (9, 1)}


def test_steps_and_uses_surgery_when_it_is_the_only_heal():
    kit = [{"id": "field_surgery", "index": 4, "name": "战地手术匣"}]
    # between floors: low HP, only the combat-only kit -> single step, never rush
    state = make_state(hp=250, bag=kit, controls=("rush", "down", "extract"))
    assert make_policy().decide(state).command == "down"
    # in an (easy) encounter: use the kit now so rushing can resume
    mob = {"name": "mob", "hp": 50, "atk": 10, "def": 2, "agi": 5}
    d = make_policy().decide(make_state(hp=180, enemy=mob, bag=kit))
    assert d.command == "use" and d.params == {"index": 4}
    # above the top-up threshold an easy fight is taken without spending a kit
    assert make_policy().decide(make_state(hp=250, enemy=mob, bag=kit)).command == "fight"
    # with a non-combat heal available, the kit is saved
    both = kit + [{"id": "coffee", "index": 5, "name": "冰美式"}]
    assert make_policy().decide(make_state(hp=250, enemy=mob, bag=both)).command == "fight"


def test_rush_check_ignores_expiring_buffs():
    policy = make_policy()
    state = make_state(hp=463, bag=KIT, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 1162
    state["run"]["effective_stats"] = {"hp": 670, "atk": 276, "def": 112, "agi": 71, "luk": 21}
    state["run"]["field_buffs"] = []
    policy.decide(state)                                   # remembers buff-free stats
    state["run"]["effective_stats"] = {"hp": 670, "atk": 276, "def": 147, "agi": 71, "luk": 21}
    state["run"]["field_buffs"] = [{"id": "field_plate"}]  # temporary DEF buff now active
    assert policy.lasting_build(state).defense == 112
    assert policy.decide(state).command == "down"           # the F1164 death situation


def test_extracts_when_healing_is_exhausted():
    state = make_state(hp=600, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 900
    assert make_policy().decide(state).command == "extract"


def test_extracts_from_risky_fight_with_no_heals_left():
    elite = {"name": "elite", "elite": True, "hp": 2700, "atk": 300, "def": 90, "agi": 55}
    d = make_policy().decide(make_state(hp=300, enemy=elite))
    assert d.command == "extract", d.reason


def test_keeps_going_while_heals_remain():
    state = make_state(hp=600, bag=[{"id": "coffee", "index": 2, "name": "冰美式"}], controls=("rush", "down", "extract"))
    state["run"]["floor"] = 40
    assert make_policy().decide(state).command == "rush"


def test_doll_makes_the_agent_greedy():
    doll = [{"id": "doll", "index": 7, "name": "替身娃娃"}]
    state = make_state(hp=600, bag=doll, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 900
    assert make_policy().decide(state).command != "extract"          # dry but doll: keep going
    elite = {"name": "elite", "elite": True, "hp": 2700, "atk": 300, "def": 90, "agi": 55}
    assert make_policy().decide(make_state(hp=300, enemy=elite, bag=doll)).command == "fight"


def test_regular_fights_are_capped_at_15_rounds():
    rng = np.random.default_rng(3)
    tank = {"hp": 10**6, "atk": 1, "def": 0, "agi": 0}
    hp, _, ehp, _, used = fight(AFFIXES, np.full(50, 600.0), np.zeros(50), tank, boss=False, rng=rng)
    assert used.max() == 15 and (ehp > 0).all()          # draw: enemy survives


def test_rush_check_ignores_temporary_leech():
    policy = make_policy()
    state = make_state(hp=582, bag=KIT, controls=("rush", "down", "extract"))
    state["run"]["floor"] = 1225
    state["player"]["secondary"]["leech"] = 0.28          # 6% set + 22% 虹吸滴管 buff
    assert policy.lasting_build(state).leech == AFFIXES.leech
