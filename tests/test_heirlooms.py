"""Tests for the blessing (闪耀祝福) helpers and how the runner uses them."""
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
