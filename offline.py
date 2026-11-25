"""Offline policy comparison against the game's own engine, run in-process.

The game author's bot repository (SlimeTROH/longniang-qqbot) contains the full game engine. This
script runs it locally on a throwaway SQLite database and plays complete runs through the normal
runner (farm.Farmer), so every policy goes through exactly the code path it uses live. Nothing is
sent to the real server.

The engine's character is replaced by ours: base stats and gear affixes as measured on the live
account (see CHARACTER), restocked with the usual loadout. The engine is an older version than the
live server (some later patches are missing), so treat results as a comparison between policies, not
as a forecast of live numbers.

    python offline.py --engine ../longniang-qqbot --runs 10 --policies threshold,markov
    python offline.py --engine ../longniang-qqbot --runs 10 --loadout "$(python -c 'import farm; print(farm.DEFAULT_LOADOUT)'),smoke=50"
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

from client import GameError
from farm import Farmer, build_parser
from model import Build

HERE = os.path.dirname(os.path.abspath(__file__))

# Live character, without temporary run effects: panel stats and gear affixes (engine perk names).
CHARACTER_STATS = {"hp": 700, "atk": 151, "def": 75, "agi": 77, "luk": 23}
CHARACTER_PERKS = {"pierce": 0.26, "prism_leech": 0.06, "heavy_strike": 0.38, "berserk": 0.075,
                   "vampire": 0.10, "high_roller": 1.9}
# The same affixes as the agent's gear model would derive them from the live warehouse.
CHARACTER_AFFIXES = Build(hp=700, atk=151, defense=75, agi=77, luk=23, pierce=0.26, leech=0.06, heavy=0.38,
                          berserk=0.075, vampire=0.10, crit_mult=1.9)
SECONDS_PER_COMMAND = 3      # virtual clock step: clears every command cooldown, far below the run timeout
GOLD = 10**11
UID, GROUP, NICK = "offline", "offline", "offline-agent"


class LocalClient:
    """ElevatorClient look-alike backed by the in-process engine (WebGame) on a private database."""

    def __init__(self, engine_dir: str, db_path: str):
        engine_dir = os.path.abspath(engine_dir)
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        from elevator_web_game import WebGame, rules  # engine import, resolved from engine_dir

        def character(_turn, *, sources=None):
            if sources is not None:  # the stat panel lists where each value comes from
                sources.update({k: [{"source": "offline character", "op": "base", "value": v, "result": v}]
                                for k, v in CHARACTER_STATS.items()})
            return dict(CHARACTER_STATS), dict(CHARACTER_PERKS)

        rules.Turn._stats = character
        self.web = WebGame(Path(db_path))
        self.now = int(time.time())
        self.state: dict | None = None

    # -- ElevatorClient interface ------------------------------------------
    def login(self, *_args, **_kwargs) -> dict:
        return {}

    def refresh(self) -> dict:
        self.state = self.web.state(UID, GROUP, NICK, now=self.now)
        self._top_up()
        return self.state

    @property
    def revision(self) -> str:
        return self.state["revision"]

    def command(self, command: str, params: dict | None = None) -> dict:
        if self.state is None:
            self.refresh()
        self.now += SECONDS_PER_COMMAND
        data = self.web.command(UID, GROUP, command, params or {}, self.revision, NICK, now=self.now)
        self.state = data["state"]
        if data.get("error"):
            raise GameError(data.get("code"), data["error"], data)
        return data

    def action(self, action: str, argument: str = "") -> dict:
        self.now += SECONDS_PER_COMMAND
        data = self.web.act(UID, GROUP, action, argument, self.revision, NICK, now=self.now)
        self.state = data["state"]
        if data.get("error"):
            raise GameError(data.get("code"), data["error"], data)
        return data

    # -- test fixture ---------------------------------------------------------
    def _top_up(self) -> None:
        """Between runs: unlimited gold and tickets, a 4-slot safe box, and an empty warehouse apart from
        supplies. Banked loot would otherwise pile up (nothing is sold offline) and the state the engine
        rebuilds after every command grows with it - 1.5 MB and 90 ms per command after seven runs."""
        if self.state.get("run"):
            return
        supplies = [k for k, v in self.web.game.catalog.items.items() if v.get("kind") == "supply"]
        marks = ",".join("?" * len(supplies))
        with self.web.game.store.transaction() as db:
            db.execute("UPDATE players SET gold=?, tickets=20, safebox=4 WHERE user_id=?", (GOLD, UID))
            db.execute(f"DELETE FROM inventory WHERE user_id=? AND item_id NOT IN ({marks})", (UID, *supplies))
            db.execute("DELETE FROM equipment WHERE user_id=?", (UID,))
        self.state = self.web.state(UID, GROUP, NICK, now=self.now)


def run_policy(policy: str, args: argparse.Namespace) -> list[dict]:
    out_dir = os.path.join(args.out, policy)
    os.makedirs(out_dir, exist_ok=True)
    db = os.path.join(out_dir, "engine.sqlite3")
    if os.path.exists(db):
        os.remove(db)
    farm_args = build_parser().parse_args([
        "--mode", "real", "--policy", policy, "--runs", str(args.runs), "--map", "mall",
        "--delay", "0", "--log-dir", out_dir, "--gold-reserve", "0", "--sell-loot", "off",
        *(["--loadout", args.loadout] if args.loadout else []),
    ])
    farmer = Farmer(farm_args, client=LocalClient(args.engine, db))
    farmer.cal.path = os.path.join(out_dir, "calibration.json")   # never touch the live calibration
    farmer.policy._affix_build, farmer.policy._loadout = CHARACTER_AFFIXES, ()   # no gear in the local db
    farmer.policy.missing_collectibles = lambda _state: set()   # the live account owns every non-rainbow item
    farmer.run()
    with open(os.path.join(out_dir, "runs.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="path to a checkout of SlimeTROH/longniang-qqbot")
    ap.add_argument("--runs", type=int, default=10, help="runs per policy")
    ap.add_argument("--policies", default="threshold,markov")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "offline"))
    ap.add_argument("--loadout", default=None, help="supplies to carry (farm.py format); default: farm.py's")
    args = ap.parse_args()
    results = {p: run_policy(p, args) for p in args.policies.split(",")}
    print("\npolicy      runs  bosses (mean +- se)  floor (mean)  steps (mean)")
    for policy, rows in results.items():
        bosses = [int(r["bosses"]) for r in rows]
        floors = [int(r["floor"]) for r in rows]
        steps = [int(r["steps"]) for r in rows]
        se = statistics.stdev(bosses) / len(bosses) ** 0.5 if len(bosses) > 1 else 0.0
        print(f"{policy:<10} {len(rows):>5}  {statistics.mean(bosses):7.1f} +- {se:4.1f}      "
              f"{statistics.mean(floors):8.0f}  {statistics.mean(steps):10.0f}")


if __name__ == "__main__":
    main()
