"""Tests for the MDP solver (mdp.py) and the policy built on it (markov_policy.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from combat import Calibration  # noqa: E402
from markov_policy import MarkovConfig, MarkovPolicy, heal_stock  # noqa: E402
from mdp import SHOP_HEALS, MdpConfig, interp_matrix, solve  # noqa: E402
from model import Build, boss_floors, enemy  # noqa: E402

# Deliberately fragile so that F100-F200 fights carry real risk (the live build is safe this shallow).
BUILD = Build(hp=666, atk=120, defense=20, agi=30, luk=21, pierce=0.26, leech=0.06, heavy=0.38,
              berserk=0.075, vampire=0.10, crit_mult=1.9)
SMALL = MdpConfig(max_floor=200, hp_points=26, boss_points=6, samples=96, kernel_stride=10,
                  lambdas=(0.0, 0.05, 0.3, 2.0))
CAL = Calibration(k_player=1.0, k_enemy=1.0, path=os.devnull)


@pytest.fixture(scope="module")
def tables():
    return solve(BUILD, "mall", SMALL)


@pytest.fixture()
def policy(tables, tmp_path):
    p = MarkovPolicy(CAL, markov=MarkovConfig(cache_dir=str(tmp_path), live_samples=150, mdp=SMALL))
    p._affix_build, p._loadout = BUILD, ()
    p.planner._tables = tables
    p._rng = np.random.default_rng(7)
    return p


def bag_of(*ids):
    return [{"id": i, "index": n + 1, "name": i, "quantity": 1} for n, i in enumerate(ids)]


def state(floor=100, hp=666, foe=None, bag=(), controls=("fight", "flee", "extract")):
    run = {
        "floor": floor, "hp": hp, "max_hp": 666, "shield": 0, "map": "mall",
        "effective_stats": {"hp": BUILD.hp, "atk": BUILD.atk, "def": BUILD.defense, "agi": BUILD.agi, "luk": BUILD.luk},
        "event": {"kind": "combat", "enemy": foe} if foe else {"kind": "ready"},
        "bag": list(bag), "field": [], "safe": [], "safe_capacity": 4,
        "field_slots": [None, None], "field_offer": None,
    }
    return {
        "run": run,
        "player": {"loadout": {}, "secondary": {"crit": 0.15, "crit_multiplier": 1.9, "pierce": 0.26, "leech": 0.06}},
        "controls": [{"command": c, "params": {}, "label": c} for c in controls],
        "atlas": {"items": []},
    }


def foe(kind, floor):
    stats = enemy(kind, floor)
    boss = kind == "boss"
    return {**stats, "max_hp": stats["hp"], "name": kind, "elite": kind != "normal", "boss": 3 if boss else 0,
            "ambush": False, "rounds_fought": 0}


# ------------------------------------------------------------------ mdp
def test_heal_items_follow_server_rules():
    e = np.array([0.30, 0.80, 1.00, 1.10])
    assert np.allclose(SHOP_HEALS["coffee"].apply(e, 700), [0.70, 1.00, 1.00, 1.10])
    assert np.allclose(SHOP_HEALS["barrier_serum"].apply(e, 700), [1.05, 1.25, 1.25, 1.25])
    assert np.allclose(SHOP_HEALS["critical_serum"].apply(e, 700), [1.00, 1.00, 1.00, 1.10])  # 90% below 35%
    assert SHOP_HEALS["triage_vial"].units(600) == pytest.approx(0.57)


def test_interp_matrix_reproduces_linear_functions():
    grid = np.linspace(0, 1.25, 11)
    x = np.array([0.0, 0.07, 0.5, 1.249, 2.0])
    w = interp_matrix(x, grid)
    assert np.allclose(w.sum(axis=1), 1)
    assert np.allclose(w @ (3 * grid + 1), 3 * np.minimum(x, 1.25) + 1)


def test_values_are_sane(tables):
    free = 0  # lambda index of free healing
    assert np.all(tables.ready >= -1e-9)
    assert tables.at(tables.ready, 1, 1.0)[free] <= len(boss_floors(SMALL.max_floor - 1)) + 1e-9
    costly = len(tables.lambdas) - 1  # with expensive heals HP itself carries value
    assert tables.at(tables.ready, 100, 1.0)[costly] > tables.at(tables.ready, 100, 0.2)[costly]
    assert np.all(np.diff(tables.ready[100][costly]) >= -0.02)                 # monotone up to MC noise
    # a single step offers every option a rush has, so rushing is never worth more
    assert np.all(tables.rush[1:200] <= tables.step[1:200] + 1e-9)


# ------------------------------------------------------------------ policy
def test_heal_stock_counts_medicine_and_field_heals():
    s = state(bag=bag_of("coffee", "coffee", "field_surgery", "adrenaline"))
    s["run"]["field"] = [{"id": "f1", "name": "速凝绷带", "quantity": 2, "healing": True}]
    assert heal_stock(s["run"]) == pytest.approx(0.4 * 2 + 1.0 + 0.28 * 2)


def test_heals_before_a_dangerous_elite(policy):
    s = state(hp=120, foe=foe("elite", 100), bag=bag_of("coffee", "triage_vial", "field_surgery"))
    assert policy.decide(s).command == "use"


def test_fights_an_elite_at_full_hp(policy):
    s = state(hp=666, foe=foe("elite", 100), bag=bag_of("coffee", "triage_vial"))
    assert policy.decide(s).command == "fight"


def test_rushes_when_healthy(policy):
    s = state(hp=666, bag=bag_of("coffee"), controls=("rush", "down", "extract"))
    assert policy.decide(s).command == "rush"


def test_low_hp_rush_defers_healing_to_the_encounter(policy):
    # the server halts a rush before any fight at HP <= 50%, and every heal also works in an encounter
    s = state(hp=150, bag=bag_of("coffee", "triage_vial"), controls=("rush", "down", "extract"))
    assert policy.decide(s).command == "rush"


def test_never_extracts_when_out_of_heals(policy):
    s = state(hp=300, controls=("rush", "down", "extract"))
    assert policy.decide(s).command in ("rush", "down")


def test_boss_window_heals_low_and_fights_high(policy):
    boss = foe("boss", 110)
    low = state(floor=110, hp=90, foe=boss, bag=bag_of("field_surgery"), controls=("fight", "extract"))
    high = state(floor=110, hp=666, foe=boss, bag=bag_of("field_surgery"), controls=("fight", "extract"))
    assert policy.decide(low).command == "use"
    assert policy.decide(high).command == "fight"


def test_shallow_floors_use_the_threshold_policy(policy):
    s = state(floor=5, hp=666, bag=bag_of("coffee"), controls=("rush", "down", "extract"))
    assert policy.decide(s).reason.startswith("rush (chain risk")


def test_planner_reuses_tables_across_small_calibration_drift(policy, tables):
    from dataclasses import replace
    drifted = replace(tables.build, taken_mult=tables.build.taken_mult * 1.004, atk=tables.build.atk * 1.01)
    assert policy.planner.tables(drifted, "mall") is tables


def test_isotonic_fit_is_monotone_and_mean_preserving():
    from markov_policy import isotonic
    y = np.array([0.0, 2.0, 1.0, 3.0, 2.5, 2.5, 5.0])
    fit = isotonic(y)
    assert np.all(np.diff(fit) >= 0)
    assert fit.sum() == pytest.approx(y.sum())
    assert np.allclose(isotonic(np.arange(5.0)), np.arange(5.0))


# ------------------------------------------------------------------ field items
def test_field_effects_follow_server_specs():
    import field
    run = {"stats": {"hp": 666, "atk": 120, "def": 20, "agi": 30, "luk": 21},
           "field_slots": [{"id": "field_lamp"}, None], "field_buffs": [], "field_charges": [{"id": "field_tape", "left": 2}]}
    live = field.live_build(run, BUILD)
    spot = field.SPECS["field_lamp"]["mods"]                       # 促销聚光灯: crit +13%, mitigation -10%
    assert live.crit_bonus == pytest.approx(spot["crit"]) and live.reduction == pytest.approx(spot["mitigation"])
    assert live.guard_hits == 2 and live.guard_reduction == field.SPECS["field_tape"]["reduction"]
    plate = field.FieldPrep("field_plate", "应急陶板", field.SPECS["field_plate"])  # defence +50% of base
    boosted, _ = field.apply(run, BUILD, plate)
    assert boosted.defense == pytest.approx(BUILD.defense + 0.5 * 20)


def test_guard_charges_cut_landed_hits():
    from dataclasses import replace
    from model import fight
    foe_ = {"hp": 10**9, "atk": 200, "def": 0, "agi": 0}
    base = replace(BUILD, agi=0, leech=0, vampire=0)
    rng = np.random.default_rng(3)
    plain = fight(base, np.full(2000, 1e6), np.zeros(2000), foe_, boss=False, rounds=1, rng=rng)[0]
    guarded = fight(replace(base, guard_hits=1, guard_reduction=0.42), np.full(2000, 1e6), np.zeros(2000), foe_,
                    boss=False, rounds=1, rng=rng)[0]
    ratio = (1e6 - guarded[guarded < 1e6]).mean() / (1e6 - plain[plain < 1e6]).mean()
    assert ratio == pytest.approx(0.58, abs=0.02)


def test_uses_a_defensive_buff_before_a_risky_fight(policy):
    s = state(floor=150, hp=380, foe=foe("elite", 150))
    s["run"]["field"] = [{"id": "field_plate", "name": "应急陶板", "quantity": 1}]
    s["run"]["stats"] = s["run"]["effective_stats"]
    decision = policy.decide(s)
    assert decision.command == "field_use" and decision.params == {"id": "field_plate"}, decision.reason


def test_keeps_buffs_for_a_safe_fight(policy):
    s = state(floor=60, hp=666, foe=foe("normal", 60))
    s["run"]["field"] = [{"id": "field_plate", "name": "应急陶板", "quantity": 1}]
    s["run"]["stats"] = s["run"]["effective_stats"]
    assert policy.decide(s).command == "fight"
