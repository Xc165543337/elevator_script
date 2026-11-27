"""Tests for the MDP solver (mdp.py) and the policy built on it (markov_policy.py)."""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from combat import Calibration  # noqa: E402
from markov_policy import MarkovConfig, MarkovPolicy, heal_stock  # noqa: E402
from mdp import SHOP_HEALS, MdpConfig, elite_pause_kernel, interp_matrix, solve  # noqa: E402
from model import LEGACY_RULES, LIVE_RULES, Build, boss_floors, enemy, fight_segment, pauses_after_segment  # noqa: E402
from policy import PolicyConfig  # noqa: E402

# Deliberately fragile so that F100-F200 fights carry real risk (the live build is safe this shallow).
BUILD = Build(hp=666, atk=120, defense=20, agi=30, luk=21, pierce=0.26, leech=0.06, heavy=0.38,
              berserk=0.075, vampire=0.10, crit_mult=1.9)
SMALL = MdpConfig(max_floor=200, hp_points=26, boss_points=6, samples=96, kernel_stride=10,
                  lambdas=(0.0, 0.05, 0.3, 2.0))
CAL = Calibration(k_player=1.0, k_enemy=1.0, path=os.devnull)


@pytest.fixture(scope="module")
def tables():
    return solve(BUILD, "mall", SMALL)


@pytest.fixture(scope="module")
def legacy_tables():
    return solve(BUILD, "mall", replace(SMALL, rules=LEGACY_RULES))


def make_policy(tables, cache_dir, rules=LIVE_RULES):
    p = MarkovPolicy(CAL, PolicyConfig(rules=rules),
                     MarkovConfig(cache_dir=str(cache_dir), live_samples=150, mdp=tables.config, background=False))
    p._affix_build, p._loadout = BUILD, ()
    p.planner._tables = tables
    p._rng = np.random.default_rng(7)
    return p


@pytest.fixture()
def policy(tables, tmp_path):
    return make_policy(tables, tmp_path)


@pytest.fixture()
def legacy_policy(legacy_tables, tmp_path):
    """The older engine of offline.py: elites fight 15 rounds without a pause."""
    return make_policy(legacy_tables, tmp_path, LEGACY_RULES)


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


def test_uses_a_defensive_buff_before_a_risky_fight(legacy_policy):
    s = state(floor=165, hp=380, foe=foe("elite", 165))
    s["run"]["field"] = [{"id": "field_plate", "name": "应急陶板", "quantity": 1}]
    s["run"]["stats"] = s["run"]["effective_stats"]
    decision = legacy_policy.decide(s)
    assert decision.command == "field_use" and decision.params == {"id": "field_plate"}, decision.reason


def test_keeps_buffs_for_a_safe_fight(policy):
    s = state(floor=60, hp=666, foe=foe("normal", 60))
    s["run"]["field"] = [{"id": "field_plate", "name": "应急陶板", "quantity": 1}]
    s["run"]["stats"] = s["run"]["effective_stats"]
    assert policy.decide(s).command == "fight"


def test_regular_kernel_separates_kills_from_draws():
    from mdp import regular_kernel
    grid = np.linspace(0, 1.25, 26)
    rng = np.random.default_rng(5)
    wall = {"hp": 10**7, "atk": 1, "def": 0, "agi": 0}               # cannot be killed in 15 rounds
    kill, draw = regular_kernel(BUILD, wall, grid, 64, rng)
    assert kill.sum() == 0 and np.allclose(draw[1:].sum(axis=1), 1)
    weak = {"hp": 5, "atk": 1, "def": 0, "agi": 0}
    kill, draw = regular_kernel(BUILD, weak, grid, 64, rng)
    assert draw.sum() == 0 and np.allclose(kill[1:].sum(axis=1), 1)


def test_tables_ignore_ice_tea_agility(policy):
    s = state(hp=666, controls=("rush", "down", "extract"))
    s["run"]["stats"] = {**s["run"]["effective_stats"]}
    s["run"]["effective_stats"] = {**s["run"]["stats"], "agi": BUILD.agi + 15}   # tea active
    assert policy._table_build(s).agi == BUILD.agi


# ------------------------------------------------------------------ smoke bombs
def test_smokes_past_a_deadly_elite(policy):
    s = state(floor=165, hp=200, foe=foe("elite", 165), bag=bag_of("smoke", "smoke"))   # boss at F170 still ahead
    decision = policy.decide(s)
    assert decision.command == "use" and decision.params == {"index": 1}, decision.reason


def test_keeps_smokes_for_a_safe_fight(policy):
    s = state(floor=60, hp=666, foe=foe("normal", 60), bag=bag_of("smoke", "smoke"))
    assert policy.decide(s).command == "fight"


def test_never_smokes_a_boss(policy):
    s = state(floor=190, hp=150, foe=foe("boss", 190), bag=bag_of("smoke"), controls=("fight", "extract"))
    assert policy.decide(s).command != "use"


def test_price_choice_counts_smokes(tables):
    # values fall as smokes get dearer (until no smoke is worth using), as in solved tables
    values = 10.0 - np.minimum(tables.smoke_prices, 1.0)
    none = tables.price_index(values, 0.0, 0)
    some = tables.price_index(values, 0.0, 5)
    assert tables.smoke_prices[none] >= 1.0                          # no smokes: a price that never uses one
    assert tables.smoke_prices[some] < max(tables.smoke_prices)


def test_single_steps_while_slow_serum_regen_is_pending(policy):
    s = state(hp=300, bag=bag_of("coffee"), controls=("rush", "down", "extract"))
    s["run"]["recovery"] = {"left": 2, "fraction": 0.16, "last_floor": 99}
    assert policy.decide(s).command == "down"


# ------------------------------------------------------------------ live fight rules (2026-11-25)
def test_fight_segment_follows_round_limits():
    elite = {"elite": True, "boss": 0, "rounds_fought": 0}
    assert fight_segment(elite, LIVE_RULES) == (1, 8) and pauses_after_segment(elite, LIVE_RULES)
    assert fight_segment({**elite, "rounds_fought": 8}, LIVE_RULES) == (9, 8)
    assert not pauses_after_segment({**elite, "rounds_fought": 8}, LIVE_RULES)
    assert fight_segment(elite, LEGACY_RULES) == (1, 15) and not pauses_after_segment(elite, LEGACY_RULES)
    assert fight_segment({"elite": False, "boss": 0}, LIVE_RULES) == (1, 15)
    assert fight_segment({"elite": True, "boss": 3, "rounds_fought": 5}, LIVE_RULES) == (6, 3)


def test_elite_pause_kernel_keeps_every_survivor():
    grid, enemy_grid = np.linspace(0, 1.25, 11), np.linspace(0.2, 1.0, 5)
    harmless = {"hp": 10**7, "atk": 1, "def": 0, "agi": 0}                # survives, never dies
    lands = elite_pause_kernel(BUILD, harmless, grid, enemy_grid, 16, np.random.default_rng(3), LIVE_RULES)
    rows = lands.sum(axis=1).reshape(len(grid), len(enemy_grid))
    assert np.allclose(rows[0], 0) and np.allclose(rows[1:], 1)


def test_heals_at_the_elite_pause_when_the_rest_is_risky(policy):
    elite = {**foe("elite", 165), "rounds_fought": 8}
    elite["hp"] = int(elite["max_hp"] * 0.6)
    s = state(floor=165, hp=100, foe=elite, bag=bag_of(*["coffee"] * 6), controls=("fight", "flee", "use"))
    decision = policy.decide(s)
    assert decision.command == "use", decision.reason


def test_fresh_elite_risk_covers_the_rounds_before_the_pause():
    from combat import estimate
    wall = {**foe("elite", 165), "hp": 10**7, "max_hp": 10**7}         # cannot be killed: risk = rounds survived
    s = state(floor=165, hp=300, foe=wall)
    live = estimate(BUILD, s, CAL, samples=2000, rng=np.random.default_rng(1), rules=LIVE_RULES)
    legacy = estimate(BUILD, s, CAL, samples=2000, rng=np.random.default_rng(1), rules=LEGACY_RULES)
    assert live.death < legacy.death


def test_policy_rules_reach_the_solver(tmp_path, legacy_tables):
    p = make_policy(legacy_tables, tmp_path, LEGACY_RULES)
    assert p.mcfg.mdp.rules == LEGACY_RULES


def test_lasting_build_ignores_a_buff_that_lowers_a_stat(policy):
    calm = state(hp=666)
    policy.lasting_build(calm)
    buffed = state(hp=666)
    buffed["run"]["field_buffs"] = [{"id": "field_clearance", "left": 2}]      # 清仓红牌: defence -20%
    buffed["run"]["effective_stats"] = {**calm["run"]["effective_stats"], "def": BUILD.defense * 0.8}
    assert policy.lasting_build(buffed).defense == BUILD.defense


# ------------------------------------------------------------------ 为了我所期待的 (wish)
def _wish_state(hp, missing_in_bag=False):
    s = state(floor=170, hp=hp, foe=foe("boss", 170), controls=("fight", "field_use"))
    s["run"]["field"] = [{"id": "field_wish", "name": "为了我所期待的", "quantity": 1}]
    if missing_in_bag:
        s["run"]["bag"] = [{"id": "prism_sun", "index": 1, "name": "熔阳刻痕", "value": 1, "can_protect": True}]
    return s


def test_wish_kills_a_boss_that_would_win(policy):
    policy.missing_collectibles = lambda _state: {"prism_sun"}
    s = _wish_state(60)
    decision = policy.decide(s)
    assert decision.command == "field_use" and decision.params == {"id": "field_wish"}, decision.reason
    s["wish_preview"] = {"enemy": "boss", "bag": 0, "safe": 0, "value": 0, "token": "0123456789abcdef"}
    confirm = policy.decide(s)
    assert confirm.command == "wish_confirm" and confirm.params == {"token": "0123456789abcdef"}


def test_wish_is_kept_for_a_safe_fight_and_never_wipes_a_missing_collectible(policy):
    policy.missing_collectibles = lambda _state: {"prism_sun"}
    assert policy.decide(_wish_state(666)).command != "field_use"
    assert policy.decide(_wish_state(60, missing_in_bag=True)).params != {"id": "field_wish"}


def test_wish_is_not_requested_again_when_no_preview_comes_back(policy):
    policy.missing_collectibles = lambda _state: set()
    s = _wish_state(60)
    assert policy.decide(s).params == {"id": "field_wish"}
    assert policy.decide(s).params != {"id": "field_wish"}            # no preview in the state: banned on this floor


def test_wish_is_not_used_when_a_full_heal_makes_the_window_safe(policy):
    policy.missing_collectibles = lambda _state: set()
    s = _wish_state(60)
    s["run"]["bag"] = bag_of("field_surgery")                       # heals to full in combat
    s["controls"].append({"command": "use", "params": {}, "label": "use"})
    assert policy.decide(s).params != {"id": "field_wish"}


# ------------------------------------------------------------------ extraction at a wall
def test_boss_win_chance_tells_a_wall_from_an_easy_boss():
    from model import boss_win_chance
    easy = {"hp": 300, "atk": 30, "def": 5, "agi": 10}
    wall = {"hp": 10**6, "atk": 5000, "def": 500, "agi": 55}
    assert boss_win_chance(BUILD, easy, samples=200) > 0.99
    assert boss_win_chance(BUILD, wall, samples=200) == 0.0


def test_extracts_between_floors_when_the_coming_bosses_are_a_wall(policy):
    s = state(hp=666, bag=bag_of("coffee"), controls=("rush", "down", "extract"))
    policy._boss_outlook = lambda _state, _floor: 0.1
    assert policy.decide(s).command == "extract"
    policy._boss_outlook = lambda _state, _floor: 2.0
    assert policy.decide(s).command != "extract"


def test_keeps_going_with_a_doll_in_the_bag(policy):
    s = state(hp=666, bag=bag_of("coffee", "doll"), controls=("rush", "down", "extract"))
    policy._boss_outlook = lambda _state, _floor: 0.1
    assert policy.decide(s).command != "extract"


def test_banks_the_bag_instead_of_fighting_a_hopeless_boss(policy):
    wall = {**foe("boss", 170), "hp": 10**6, "max_hp": 10**6, "atk": 5000}
    s = state(floor=170, hp=666, foe=wall, bag=bag_of("coffee", "smoke"), controls=("fight", "use", "extract"))
    assert policy.decide(s).command == "extract"
    s["controls"] = [c for c in s["controls"] if c["command"] != "extract"]
    assert policy.decide(s).command == "use"                               # smoke: ends the run as an extraction


# ------------------------------------------------------------------ rushing through elites (自动打精英)
def test_rushing_through_elites_never_beats_stopping_and_is_free_when_shallow(tables):
    assert np.all(tables.rush_elite[1:200] <= tables.rush[1:200] + 1e-6)
    assert np.allclose(tables.rush_elite[1:60], tables.rush[1:60], atol=1e-3)   # shallow: elites are no threat
    assert (tables.rush[100:200] - tables.rush_elite[100:200]).max() > 0.05     # deep: a stop to smoke / heal pays


def _rush_ready(flag):
    s = state(floor=40, hp=666, bag=bag_of("coffee"), controls=("rush", "down", "extract"))
    s["player"]["rush_auto_elite"] = flag
    return s


@pytest.mark.parametrize("loss, auto", [(0.001, True), (0.01, False)])
def test_rushes_through_elites_within_the_tolerance(tables, tmp_path, loss, auto):
    worse = replace(tables, rush_elite=tables.rush_elite.copy())
    worse.rush_elite[40] -= loss
    policy = make_policy(worse, tmp_path)
    switch = policy.decide(_rush_ready(not auto))
    assert (switch.command, switch.params) == ("rush_settings", {"auto_elite": auto}), switch.reason
    assert policy.decide(_rush_ready(auto)).command == "rush"


def test_rush_elites_option_reaches_the_solver(tmp_path):
    p = MarkovPolicy(CAL, PolicyConfig(rush_elites=False), MarkovConfig(cache_dir=str(tmp_path)))
    assert not p.mcfg.mdp.rush_elites


# ------------------------------------------------------------------ table cache and background solves
TINY = MdpConfig(max_floor=40, hp_points=11, boss_points=4, samples=16, kernel_stride=10, lambdas=(0.0, 1.0),
                 smoke_prices=(0.0, 1000.0), pause_samples=8, boss_iterations=5)


def test_bucket_is_conservative_and_shared_by_nearby_builds():
    from markov_policy import bucket
    a = bucket(BUILD, 0.05)
    assert a.atk <= BUILD.atk and a.hp <= BUILD.hp and a.taken_mult >= BUILD.taken_mult
    assert a.atk > BUILD.atk / 1.05 and a.leech <= BUILD.leech
    near = replace(a, atk=a.atk * 1.03, hp=a.hp + 3, taken_mult=a.taken_mult * 0.98)
    assert bucket(near, 0.05) == a                          # within one grid step: one cached solve
    assert bucket(a, 0.05) == a                             # idempotent
    assert bucket(BUILD, 0.0) == BUILD


def test_background_solve_keeps_playing_on_old_tables(tmp_path):
    from markov_policy import Planner
    cfg = MarkovConfig(cache_dir=str(tmp_path), mdp=TINY, background=True)
    planner = Planner(cfg)
    assert planner.tables(BUILD, "mall") is None            # nothing yet: the policy uses its threshold rules
    planner.wait()
    first = planner.tables(BUILD, "mall")
    assert first is not None and first.config == TINY
    stronger = replace(BUILD, atk=BUILD.atk * 1.5)
    assert planner.tables(stronger, "mall") is first        # old tables while the new ones solve
    planner.wait()
    assert planner.tables(stronger, "mall") is not first
    assert planner.tables(stronger, "subway") is None       # never another map's tables


def test_cache_keeps_the_most_recently_used_solves(tmp_path):
    from markov_policy import Planner
    planner = Planner(MarkovConfig(cache_dir=str(tmp_path), mdp=TINY, background=False, cache_max_files=2))
    for k in range(3):
        planner.tables(replace(BUILD, atk=BUILD.atk * (1.3 ** k)), "mall")
    assert len(list(tmp_path.glob("mdp_*.npz"))) == 2
    reloaded = Planner(planner.cfg)
    assert reloaded.tables(replace(BUILD, atk=BUILD.atk * 1.3 ** 2), "mall") is not None   # from disk


# ------------------------------------------------------------------ gold objective
WEAK = replace(BUILD, hp=400, defense=5, atk=90, leech=0.0, vampire=0.0)   # outmatched from about F70


@pytest.fixture(scope="module")
def gold_tables():
    return solve(WEAK, "mall", replace(SMALL, gold=True, boss_value=0.2))


def test_loot_curve_interpolates_and_extends_with_the_last_slope():
    from mdp import LOOT_CURVES, loot_at
    assert loot_at("mall", 0) == 0.0 and loot_at("mall", 100) == pytest.approx(0.75)
    assert loot_at("mall", 150) == pytest.approx((0.75 + 1.71) / 2)
    (x1, y1), (x2, y2) = LOOT_CURVES["mall"][-2:]
    assert loot_at("mall", x2 + 100) == pytest.approx(y2 + 100 * (y2 - y1) / (x2 - x1))
    assert loot_at("unknown", 300) == loot_at("mall", 300)           # unmeasured maps: the mall curve
    assert loot_at("zenith", 100) > loot_at("mall", 100)             # richer maps find more per floor


def test_gold_tables_bank_at_hopeless_bosses_and_never_go_below_extracting(gold_tables):
    assert np.all(gold_tables.ready[1:200, :, 1:] >= -1e-6)                  # extracting is always possible
    costly = len(gold_tables.lambdas) - 1
    zeros = [int((gold_tables.boss[g][0][costly][1:] <= 1e-9).sum()) for g in (30, 90, 150, 190)]
    assert zeros[0] == 0 and zeros == sorted(zeros) and zeros[-1] > 50       # deeper bosses: bank more often


def test_gold_objective_adds_the_loot_found_and_keeps_boss_mode_unchanged(tables):
    gold = solve(BUILD, "mall", replace(SMALL, gold=True))
    full = np.searchsorted(tables.grid, 1.0)
    from mdp import loot_at
    # free heals, a build that survives the horizon: the gold run is worth the loot on top of the bosses
    assert gold.ready[1][0][full] - tables.ready[1][0][full] == pytest.approx(loot_at("mall", 201), abs=0.05)
    again = solve(BUILD, "mall", SMALL)
    assert np.array_equal(again.ready, tables.ready)


def test_gold_policy_extracts_instead_of_fighting_a_hopeless_boss(gold_tables, tmp_path):
    p = make_policy(gold_tables, tmp_path)
    p.mcfg = replace(p.mcfg, extract_below=0.0)                            # the table decides, not the outlook
    p._affix_build = WEAK
    loot = [{"id": "r05", "index": 9, "name": "红", "value": 48_000, "quantity": 30}]
    s = state(floor=190, hp=400, foe=foe("boss", 190), bag=loot, controls=("fight", "extract"))
    s["run"]["max_hp"] = 400
    s["run"]["effective_stats"] = {"hp": 400, "atk": 90, "def": 5, "agi": WEAK.agi, "luk": WEAK.luk}
    d = p.decide(s)
    assert d.command == "extract", d.reason


def test_bag_at_risk_counts_the_bag_only():
    from markov_policy import bag_at_risk
    run = {"bag": [{"value": 48_000, "quantity": 10}, {"value": 2_000}], "safe": [{"value": 48_000}]}
    assert bag_at_risk(run) == pytest.approx(0.482)
