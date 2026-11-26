"""Tests for the trial planner's pure parts (trial.py): outcome values and the live replay's checks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

import pytest  # noqa: E402

import trial  # noqa: E402


def run_state(hp=300, max_hp=400, bag=(), field=(), enemy=None, floor=10):
    return {"run": {"floor": floor, "hp": hp, "max_hp": max_hp, "bag": list(bag), "field": list(field),
                    "event": {"kind": "combat", "enemy": enemy} if enemy else {"kind": "ready"}},
            "controls": [], "prophecy": {"week": "2026-W48"}}


def test_value_counts_hp_and_what_is_left_to_spend():
    bare = trial.value(run_state())
    assert bare == 300
    coffee = trial.value(run_state(bag=[{"id": "coffee", "index": 1}]))
    assert coffee == pytest.approx(300 + trial.HEAL_WORTH * 0.40 * 400)
    buff = trial.value(run_state(field=[{"id": "field_foil", "quantity": 2}]))
    assert buff == 300 + 2 * trial.ITEM_WORTH
    assert trial.value({"run": None}) == trial.DEAD


def test_signature_ignores_order_and_flags_an_ended_run():
    a = trial.signature(run_state(bag=[{"id": "b"}, {"id": "a"}]))
    b = trial.signature(run_state(bag=[{"id": "a"}, {"id": "b"}]))
    assert a == b
    assert trial.signature({"run": None}) == {"over": True}


class FakeLive:
    """Plays a scripted list of states; `drift` changes the HP from that move on."""

    def __init__(self, states, drift_at=None):
        self.states, self.drift_at, self.i = states, drift_at, 0
        self.state = {"run": None, "prophecy": {"week": "2026-W48"}, "controls": []}
        self.sent = []

    def call(self, command, params):
        self.sent.append(command)
        if command == "prophecy_start":
            self.state = self.states[0]
            return {}
        self.i += 1
        state = self.states[min(self.i, len(self.states) - 1)]
        if self.drift_at is not None and self.i >= self.drift_at and state.get("run"):
            state = {**state, "run": {**state["run"], "hp": state["run"]["hp"] - 1}}
        self.state = state
        return {}


def _route(states):
    route = {"week": "2026-W48", "moves": [{"floor": s["run"]["floor"] if s.get("run") else 0, "command": "down",
                                            "params": {}, "expect": trial.signature(s)} for s in states[1:]]}
    return json.loads(json.dumps(route))   # as replay reads it from route.json


def test_replay_follows_a_matching_route_to_the_end():
    field = [{"id": "field_foil", "quantity": 1}]          # tuples vs JSON lists once broke this
    states = [run_state(floor=f, field=field) for f in (1, 2, 3)] + [{"run": None, "last_departure": {"floor": 3}}]
    live = FakeLive(states)
    out = trial.replay(live, _route(states), log=lambda _m: None)
    assert out["diverged"] is None and live.sent == ["prophecy_start", "down", "down", "down"]


def test_replay_stops_following_on_the_first_difference(monkeypatch):
    states = [run_state(floor=f) for f in (1, 2, 3, 4)]
    live = FakeLive(states, drift_at=2)

    class Stop(Exception):
        pass

    def no_policy(*_a, **_k):   # the fallback policy would keep playing: stop the test there
        raise Stop

    monkeypatch.setattr(trial.TrialPolicy, "decide", no_policy)
    with pytest.raises(Stop):
        trial.replay(live, _route(states), log=lambda _m: None)
    assert live.sent == ["prophecy_start", "down", "down"]


def test_replay_refuses_another_week_or_a_running_run():
    states = [run_state(floor=1)]
    route = {**_route(states), "week": "2026-W47"}
    with pytest.raises(SystemExit):
        trial.replay(FakeLive(states), route, log=lambda _m: None)
    busy = FakeLive(states)
    busy.state = states[0]
    with pytest.raises(SystemExit):
        trial.replay(busy, _route(states), log=lambda _m: None)
