"""Tests for the blessing (闪耀祝福) helpers and the runner's server-side settings."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import heirlooms  # noqa: E402
from farm import Farmer, build_parser  # noqa: E402

HEIRLOOMS = {
    "sets": {"A": {"name": "晨星"}, "B": {"name": "回描"}, "H": {"name": "压舱"}},
    "pieces": {"A01": {"name": "晨星一", "set": "A"}, "A02": {"name": "晨星二", "set": "A"},
               "B01": {"name": "回描一", "set": "B"}},
    "owned": {"A01": 1},
    "slots": ["A01", None, None, None],
    "focus": ["A", "B"],
}


def test_focus_by_number_follows_server_order():
    assert heirlooms.resolve_focus(HEIRLOOMS, "1,3") == ["A", "H"]
    assert heirlooms.resolve_focus(HEIRLOOMS, "1 3") == ["A", "H"]


def test_focus_by_id_and_default():
    assert heirlooms.resolve_focus(HEIRLOOMS, "H,B") == ["H", "B"]
    assert heirlooms.resolve_focus(HEIRLOOMS, None) == ["A", "B"]          # the last choice


@pytest.mark.parametrize("wanted", ["1", "1,1", "1,9", "A,Z"])
def test_focus_needs_two_different_known_sets(wanted):
    with pytest.raises(heirlooms.FocusError):
        heirlooms.resolve_focus(HEIRLOOMS, wanted)


def test_no_blessings_on_older_servers():
    assert heirlooms.resolve_focus(None, "1,2") is None
    assert heirlooms.collection_lines(None) == []


def test_new_pieces_lists_each_gained_copy():
    after = Counter({"A01": 2, "B01": 1})
    assert heirlooms.new_pieces(heirlooms.owned(HEIRLOOMS), after, HEIRLOOMS) == ["晨星一", "回描一"]


def test_collection_lines():
    lines = heirlooms.collection_lines(HEIRLOOMS)
    assert lines[0] == "1. 晨星: 1/2 collected, 1 worn (pursued)"
    assert lines[2] == "3. 压舱: 0/0 collected, 0 worn"


class RecordingClient:
    def __init__(self, state):
        self.state = state
        self.sent = []

    @property
    def revision(self):
        return "r"

    def command(self, command, params=None):
        self.sent.append((command, params))
        return {"state": self.state}


def test_real_start_sends_the_pursued_sets(tmp_path):
    args = build_parser().parse_args(["--mode", "real", "--delay", "0", "--log-dir", str(tmp_path), "--focus", "1,3"])
    client = RecordingClient({"supplies": [{"id": "coffee", "quantity": 5}], "heirlooms": HEIRLOOMS})
    farmer = Farmer(args, client=client)
    farmer.focus = farmer.pursuit()
    farmer.start_run({"coffee": 50})
    assert client.sent == [("start", {"map_id": "mall", "focus": ["A", "H"],
                                      "supplies": [{"id": "coffee", "quantity": 5}]})]


def test_pending_choice_is_sent_during_the_run(tmp_path):
    args = build_parser().parse_args(["--mode", "real", "--delay", "0", "--log-dir", str(tmp_path)])
    client = RecordingClient({"heirlooms": {**HEIRLOOMS, "needs_focus": True}})
    farmer = Farmer(args, client=client)
    farmer.focus = farmer.pursuit()
    farmer.ensure_pursuit()
    assert client.sent == [("pursue", {"focus": ["A", "B"]})]


def test_rush_settings_stop_before_elites(tmp_path):
    args = build_parser().parse_args(["--delay", "0", "--log-dir", str(tmp_path)])
    player = {"rush_floors": 10, "rush_skip_valuable": True, "rush_event_mode": "pause", "rush_auto_elite": True}
    client = RecordingClient({"player": player})
    Farmer(args, client=client).ensure_rush_settings()
    assert client.sent == [("rush_settings", {"auto_elite": False})]
    older = RecordingClient({"player": {"rush_floors": 5, "rush_skip_valuable": True}})
    Farmer(args, client=older).ensure_rush_settings()
    assert older.sent == [("rush_settings", {"floors": 10})]


def test_full_b06_set_sets_pierce_cap_and_beat():
    worn = {"slots": ["B06-1", "B06-2", "B06-3", "B06-4"], "pieces": {}}
    assert heirlooms.build_effects(worn) == {"pierce_cap": 0.87, "beat": True}
    assert heirlooms.build_effects({"slots": ["B06-1", "B06-2", None, "B08-1"]}) == {"pierce_cap": 0.87}
    assert heirlooms.build_effects(None) == {}


def test_beat_rounds_cut_damage_taken():
    import numpy as np
    from dataclasses import replace
    from model import Build, fight
    base = Build(hp=1000, atk=1, defense=0, agi=0, luk=0)
    foe = {"hp": 10**6, "atk": 100, "def": 0, "agi": 0}
    taken = {}
    for beat in (False, True):
        hp, *_ = fight(replace(base, beat=beat), np.full(4000, 1000.0), np.zeros(4000), foe, boss=False,
                       rounds=4, rng=np.random.default_rng(0))
        taken[beat] = 1000 - hp.mean()
    assert taken[True] / taken[False] == pytest.approx((3 + 0.65) / 4, rel=0.03)


def test_battle_enemy_applies_the_depth_shred():
    from combat import battle_enemy
    state = {"player": {"secondary": {"condition": {"depth_shred": 0.25}}}}
    assert battle_enemy(state, {"def": 80, "hp": 5})["def"] == 60
    assert battle_enemy({"player": {}}, {"def": 80})["def"] == 80


def test_depth_shred_matches_the_live_values():
    from model import depth_shred
    assert depth_shred(50, 0.6) == 0
    assert depth_shred(51, 0.6) == pytest.approx(0.0069767441860465115)
    assert depth_shred(63, 0.6) == pytest.approx(0.07959183673469387)
    assert depth_shred(5000, 0.6) < 0.6 and depth_shred(900, 0.0) == 0


def test_connect_waits_out_a_restarting_server(tmp_path):
    import requests

    class Flaky(RecordingClient):
        calls = 0

        def login(self):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise requests.ConnectionError("non-JSON response (HTTP 502)")

        def refresh(self):
            return self.state

    args = build_parser().parse_args(["--delay", "0", "--log-dir", str(tmp_path)])
    farmer = Farmer(args, client=Flaky({}))
    farmer.connect(attempts=5, wait=0)
    assert Flaky.calls == 3
