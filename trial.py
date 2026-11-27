"""Planner for the weekly fixed-seed trial 「来自群星的寓言集」 (Starlight Fables, the game's prophecy.py).

Every entrant of an ISO week (Beijing time) gets the same build, supplies and floors, and every random
roll of the trial comes from `Random(f"prophecy:world:{week}:F{floor}:{tag}")` - encounters, loot,
events, fights (tag `combat:{rounds fought}`) and flee attempts alike. The same moves from the same
state therefore always give the same result. This module plays the trial in the game's own engine
(a local checkout of the bot repository, as offline.py does), searches for the deepest route there,
and replays that route on the live server, checking after every step that both agree.

Score (prophecy.record_score): floors cleared, then floor reached, then extracted, then loot value.
A boss floor only counts as cleared once its boss is dead, and bosses cannot be fled, so the bosses
decide the score. Everything between them is played to arrive at each boss with as much as possible.

    python trial.py plan --engine ../longniang-qqbot --out logs/trial      # search, write the route
    python trial.py replay --route logs/trial/route.json --i-understand-live   # play it live
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from client import GameError
from combat import Calibration, estimate
from offline import UID, LocalClient
from policy import BossFarmPolicy, Decision, PolicyConfig

HERE = os.path.dirname(os.path.abspath(__file__))
NICKNAME = "trial-planner"


# ---------------------------------------------------------------------------------------------- engine
class TrialEngine:
    """The trial in a local copy of the game engine, with cheap snapshots of the run."""

    def __init__(self, engine_dir: str, db_path: str):
        if os.path.exists(db_path):
            os.remove(db_path)
        self.client = LocalClient(engine_dir, db_path)
        self.client.refresh()
        self.prophecy = sys.modules["_elevator_web_rules.prophecy"]
        self.store = self.client.web.game.store
        required = self.prophecy.required_collections(self.client.web.game.catalog)
        with self.store.transaction() as db:   # entry needs the full gold / red collection
            db.executemany("INSERT OR REPLACE INTO collections(user_id,item_id,count,first_got_at) VALUES(?,?,1,?)",
                           [(UID, key, self.client.now) for key in required])
        self.client.refresh()

    @property
    def state(self) -> dict:
        return self.client.state

    def start(self) -> None:
        self.client.command("prophecy_start", {"nickname": NICKNAME})

    def step(self, decision: Decision) -> dict:
        return self.client.command(decision.command, decision.params)

    def snapshot(self) -> tuple:
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM active_runs WHERE user_id=?", (UID,)).fetchone()
        return (dict(row) if row else None, self.client.now)

    def restore(self, snap: tuple) -> None:
        row, now = snap
        with self.store.transaction() as db:
            db.execute("DELETE FROM active_runs WHERE user_id=?", (UID,))
            if row:
                db.execute(f"INSERT INTO active_runs({','.join(row)}) VALUES({','.join('?' * len(row))})",
                           tuple(row.values()))
        self.client.now = now
        self.client.refresh()


# ---------------------------------------------------------------------------------------------- policy
class TrialPolicy(BossFarmPolicy):
    """The farm policy for everything between decisions the planner makes itself: events, temp gear,
    chests, field items. Differences: flee whatever can be fled (nothing heals on a kill in this build,
    a ring shield refills on every floor), step floor by floor (a rush fights every monster), never
    extract on its own, and nothing to protect (the trial awards nothing)."""

    def __init__(self):
        super().__init__(Calibration(path=os.devnull), PolicyConfig(extract_when_dry=False, rush_elites=False,
                                                                     start_adrenaline=0, greedy_with_doll=False))
        self.star_choice: str | None = None   # set by the planner before a star cache

    @staticmethod
    def missing_collectibles(state: dict) -> set[str]:
        return set()

    def estimate(self, state: dict):
        """The farm policy's Monte Carlo estimate, seeded from the state so that plans are reproducible."""
        run = state["run"]
        enemy = run["event"]["enemy"]
        rng = np.random.default_rng([run["floor"], int(enemy.get("rounds_fought") or 0), run["hp"], int(enemy["hp"])])
        return estimate(self.current_build(state), state, self.cal, samples=self.cfg.samples, rng=rng,
                        rules=self.cfg.rules)

    def _decide(self, state: dict) -> Decision:
        run = state["run"]
        picks = [c for c in state.get("controls") or [] if c["command"] == "prophecy_pick"]
        if picks and not run.get("field_offer"):
            wanted = self.star_choice or self._default_star(run)
            pick = next((c for c in picks if c["params"].get("id") == wanted), picks[0])
            return Decision("prophecy_pick", pick["params"], f"star {pick['params'].get('id')}")
        return super()._decide(state)

    @staticmethod
    def _default_star(run: dict) -> str:
        rows = {r["id"]: r for r in run.get("star_relics") or [] if r.get("offered")}
        # relative gain: HP and defence keep us alive at bosses, attack ends them sooner
        stats = run["effective_stats"]
        score = {k: r["gain"] / max(1.0, stats.get({"lens": "atk", "anchor": "def", "heart": "hp"}[k], 1))
                 for k, r in rows.items()}
        return max(score, key=score.get)

    def _fight(self, state: dict, ctr: dict) -> Decision:
        enemy = state["run"]["event"]["enemy"]
        if "flee" in ctr and not enemy.get("boss"):
            return Decision("flee", {}, "flee: kills give nothing back in the trial")
        return super()._fight(state, ctr)

    def _advance(self, state: dict, ctr: dict) -> Decision:
        heal = self._best_heal(state["run"], in_combat=False, force=False)
        if heal and heal.command == "field_use":
            return heal
        if "down" in ctr:
            return Decision("down", {}, "step")
        return super()._advance(state, ctr)


# ---------------------------------------------------------------------------------------------- planner
DEAD = float("-inf")
HEAL_WORTH = 0.6        # an unused heal is worth this share of the HP it restores
ITEM_WORTH = 40.0       # an unused buff / shield / grenade, in HP


def signature(state: dict) -> dict:
    """What the replay compares after every move: the live run must match the planned one exactly.
    JSON-stable (lists, not tuples): the route stores it as JSON."""
    run = state.get("run")
    if not run:
        return {"over": True}
    enemy = (run.get("event") or {}).get("enemy") or {}
    return {"floor": run["floor"], "hp": run["hp"], "max_hp": run["max_hp"], "kind": (run.get("event") or {}).get("kind"),
            "enemy_hp": enemy.get("hp"), "bag": sorted(it.get("id") for it in run.get("bag") or []),
            "field": sorted([it.get("id"), it.get("quantity", 0)] for it in run.get("field") or [])}


def value(state: dict) -> float:
    """Worth of a state for comparing outcomes of the same decision: HP plus what is left to spend."""
    run = state.get("run")
    if not run:
        return DEAD
    worth = float(run["hp"])
    for heal, _ in TrialPolicy._bag_heals(run, in_combat=True):
        worth += HEAL_WORTH * heal * run["max_hp"]
    for item in run.get("field") or []:
        qty = item.get("quantity", 0)
        if item.get("healing"):
            heal = TrialPolicy._field_heals({"field": [item]})
            worth += qty * HEAL_WORTH * (heal[0][0] if heal else 0.2) * run["max_hp"]
        else:
            worth += qty * ITEM_WORTH
    worth += ITEM_WORTH * sum(1 for it in run.get("bag") or [] if it.get("id") in ("grenade", "shield"))
    return worth


def enemy_of(state: dict) -> dict | None:
    run = state.get("run")
    return ((run or {}).get("event") or {}).get("enemy")


@dataclass
class Node:
    snap: tuple
    moves: list
    items: int            # item uses since the last flee / fight
    score: float = 0.0
    hp: int = -1
    key: str = ""


class Planner:
    """Plays the trial in the engine, deciding encounters by exact search."""

    def __init__(self, engine: TrialEngine, policy: TrialPolicy | None = None, log=print,
                 beam: int = 4, boss_beam: int = 6, max_items: int = 2, star_horizon: int = 2):
        self.engine, self.policy, self.log = engine, policy or TrialPolicy(), log
        self.beam, self.boss_beam, self.max_items, self.star_horizon = beam, boss_beam, max_items, star_horizon
        self.route: list[dict] = []

    # -- moves ------------------------------------------------------------------------------------
    def _do(self, decision: Decision, record: list | None) -> bool:
        """Apply a move; False if the engine refuses it."""
        floor = self.engine.state["run"]["floor"]
        try:
            data = self.engine.step(decision)
        except GameError as err:
            if err.code != "invalid":
                raise
            self.policy.ban(floor, decision, message=str(err))
            return False
        if record is not None:
            record.append({"floor": floor, "command": decision.command, "params": decision.params,
                           "reason": decision.reason, "expect": signature(self.engine.state),
                           "summary": (data.get("result") or {}).get("summary", "")})
        return True

    def _options(self, state: dict, items_allowed: bool) -> list[Decision]:
        run = state["run"]
        ctr = {c["command"] for c in state.get("controls") or []}
        out = [Decision(c, {}, c) for c in ("flee", "fight") if c in ctr]
        if not items_allowed:
            return out
        seen = set()
        for heal, item in TrialPolicy._bag_heals(run, in_combat=True):
            if item["id"] not in seen and heal * run["max_hp"] > 0.15 * (run["max_hp"] - run["hp"]) and run["hp"] < run["max_hp"]:
                seen.add(item["id"])
                out.append(Decision("use", {"index": item["index"]}, f"use {item.get('name', item['id'])}"))
        grenade = TrialPolicy._bag_item(run, "grenade")
        if grenade:
            out.append(Decision("use", {"index": grenade["index"]}, "grenade"))
        for item in run.get("field") or []:
            if item.get("quantity", 0) > 0 and item["id"] not in seen:
                seen.add(item["id"])
                out.append(Decision("field_use", {"id": item["id"]}, f"field {item.get('name', item['id'])}"))
        return out

    # -- encounters ---------------------------------------------------------------------------------
    def encounter(self) -> bool:
        """Resolve the encounter at hand by beam search; False if every line loses the run."""
        engine = self.engine
        start_floor = engine.state["run"]["floor"]
        boss = bool(enemy_of(engine.state).get("boss"))
        width = self.boss_beam if boss else self.beam
        root = Node(engine.snapshot(), [], 0)
        root_hp = engine.state["run"]["hp"]
        best_leaf: Node | None = None
        frontier = [root]
        quick = not boss            # most monsters: a clean flee (or a clean kill) needs no search
        for _layer in range(60):
            children: list[Node] = []
            for node in frontier:
                engine.restore(node.snap)
                state = engine.state
                limit = self.max_items if boss else 1
                options = self._options(state, items_allowed=not quick and node.items < limit)
                for decision in options:
                    engine.restore(node.snap)
                    moves: list = []
                    if not self._do(decision, moves):
                        continue
                    after = engine.state
                    child = Node(engine.snapshot(), node.moves + moves,
                                 node.items + 1 if decision.command in ("use", "field_use") else 0,
                                 key=json.dumps(signature(after), sort_keys=True))
                    enemy = enemy_of(after)
                    resolved = not after.get("run") or enemy is None or after["run"]["floor"] != start_floor
                    if resolved:
                        child.score = value(after)
                        child.hp = after["run"]["hp"] if after.get("run") else -1
                        if best_leaf is None or child.score > best_leaf.score:
                            best_leaf = child
                    else:
                        progress = 1 - enemy["hp"] / max(1, enemy.get("max_hp") or enemy["hp"])
                        child.score = value(after) + 2 * progress * after["run"]["max_hp"]
                        children.append(child)
            if quick:
                if best_leaf is not None and best_leaf.score > DEAD and best_leaf.hp >= root_hp:
                    break                   # no loss at all: take it
                quick = False
                frontier, children, best_leaf = [root], [], None
                continue
            unique: dict = {}
            for child in sorted(children, key=lambda n: -n.score):
                unique.setdefault(child.key, child)
            frontier = list(unique.values())[:width]
            if not frontier:
                break
        if best_leaf is None or best_leaf.score == DEAD:
            engine.restore(root.snap)
            return False
        engine.restore(best_leaf.snap)
        self.route.extend(best_leaf.moves)
        return True

    # -- the run --------------------------------------------------------------------------------------
    def _star(self) -> None:
        """Pick the star relic whose line reaches deepest within the next `star_horizon` bosses."""
        engine = self.engine
        offered = [r["id"] for r in engine.state["run"].get("star_relics") or [] if r.get("offered")]
        root = engine.snapshot()
        results = {}
        for key in offered:
            engine.restore(root)
            probe = Planner(engine, self.policy, log=lambda _m: None, beam=self.beam, boss_beam=self.boss_beam,
                            max_items=self.max_items, star_horizon=0)
            self.policy.star_choice = key
            floor = engine.state["run"]["floor"]
            bosses_left = self.star_horizon
            target = floor
            while bosses_left:
                target = next(g for g in range(target + 1, 2001) if self._is_boss(g))
                bosses_left -= 1
            probe.run(until_floor=target + 1)
            state = engine.state
            reached = state["run"]["floor"] if state.get("run") else probe.last_floor
            results[key] = (reached, value(state))
        best = max(results, key=results.get)
        engine.restore(root)
        self.policy.star_choice = best
        self.log(f"F{engine.state['run']['floor']} star {best} ({', '.join(f'{k}: F{r[0]} {r[1]:.0f}' for k, r in results.items())})")

    def _is_boss(self, floor: int) -> bool:
        return bool(self.engine.prophecy.boss_number(floor))

    @property
    def last_floor(self) -> int:
        return self.route[-1]["floor"] if self.route else 0

    def run(self, until_floor: int = 10**9) -> None:
        """Play until the run ends or reaches until_floor."""
        engine = self.engine
        while engine.state.get("run") and engine.state["run"]["floor"] < until_floor:
            state = engine.state
            run = state["run"]
            if enemy_of(state) is not None and not run.get("field_offer"):
                if self.encounter():
                    continue
                # every line dies: on a boss floor extracting still counts the floors below it
                self.log(f"F{run['floor']}: no surviving line against {enemy_of(state).get('name')} -> extract")
                if not self._do(Decision("extract", {}, "no surviving line"), self.route):
                    self._do(Decision("fight", {}, "no surviving line, cannot extract"), self.route)
                continue
            if self.star_horizon and not run.get("field_offer") and any(
                    c["command"] == "prophecy_pick" for c in state.get("controls") or []):
                self._star()
            decision = self.policy.decide(state)
            self.policy.star_choice = None if decision.command == "prophecy_pick" else self.policy.star_choice
            self._do(decision, self.route)
            if run["floor"] % 50 == 0 and decision.command == "down":
                self.log(f"F{run['floor']} hp {run['hp']}/{run['max_hp']} value {value(state):.0f}")


# ---------------------------------------------------------------------------------------------- playing
@dataclass
class Outcome:
    floor: int
    alive: bool
    moves: list = field(default_factory=list)


def play(engine: TrialEngine, policy: TrialPolicy, max_steps: int = 100_000, log=print) -> Outcome:
    """Play with the policy until the run ends; returns the moves taken."""
    moves = []
    for _ in range(max_steps):
        run = engine.state.get("run")
        if not run:
            break
        decision = policy.decide(engine.state)
        try:
            data = engine.step(decision)
        except GameError as err:
            if err.code != "invalid":
                raise
            policy.ban(run["floor"], decision, message=str(err))
            continue
        moves.append({"floor": run["floor"], "command": decision.command, "params": decision.params,
                      "reason": decision.reason})
        result = data.get("result") or {}
        if decision.command not in ("down", "flee") or run["floor"] % 50 == 0:
            log(f"F{run['floor']} hp {run['hp']}/{run['max_hp']} -> {decision.command} {decision.params}: "
                f"{decision.reason} | {result.get('summary', '')}")
    departure = (engine.state.get("last_departure") or {})
    return Outcome(floor=moves[-1]["floor"] if moves else 0, alive=bool(departure.get("extracted")), moves=moves)


# ---------------------------------------------------------------------------------------------- replay
class LiveTrial:
    """The live server, paced and retried like farm.Farmer.call."""

    def __init__(self, client, delay: float, log):
        self.client, self.delay, self.log = client, delay, log

    @property
    def state(self) -> dict:
        return self.client.state

    def call(self, command: str, params: dict) -> dict:
        """Send one move. After a network error the move may have reached the server anyway: resend it
        only if the state's revision did not move (a second 压舱盾匣 once cost 9% HP that way)."""
        import requests
        before = self.client.revision
        for attempt in range(40):
            try:
                time.sleep(self.delay)
                if attempt and self._applied(before):
                    return {}
                return self.client.command(command, params)
            except GameError as err:
                if err.code == "cooldown":
                    time.sleep(1.0)
                    continue
                if err.code in ("invalid", "forbidden") or attempt >= 3:
                    raise
                self.log(f"retry {command} after {err}")
                self.client.refresh()
            except (requests.RequestException, ValueError) as err:
                self.log(f"network error ({err}); backoff")
                time.sleep(min(60, 2 ** attempt))
                try:
                    self.client.refresh()
                except (requests.RequestException, ValueError):
                    pass
        raise RuntimeError(f"command {command} failed repeatedly")

    def _applied(self, before: str) -> bool:
        import requests
        try:
            self.client.refresh()
        except (requests.RequestException, ValueError):
            return False
        return self.client.revision != before


def replay(live: LiveTrial, route: dict, log=print, trace=None, finish_with_policy: bool = False) -> dict:
    """Play the planned moves live. On the first difference from the plan, stop and leave the run as it
    is (the run expires after 30 idle minutes), or finish it with the plain policy. Stopping keeps the
    run recoverable: the local engine can rebuild the live state and plan on from there, which the
    policy's live decisions (using the account's own gear) would make hard to reproduce."""
    week = (live.state.get("prophecy") or {}).get("week")
    if week != route["week"]:
        raise SystemExit(f"the route is for {route['week']}, the server is in {week}: plan again")
    if live.state.get("run"):
        raise SystemExit("a run is in progress on the account: finish it first")
    live.call("prophecy_start", {})
    diverged = None
    for i, move in enumerate(route["moves"]):
        data = live.call(move["command"], move["params"])
        got = json.loads(json.dumps(signature(live.state)))
        if trace is not None:
            trace.write(json.dumps({"i": i, "move": move, "got": got,
                                    "summary": (data.get("result") or {}).get("summary", "")}, ensure_ascii=False) + "\n")
        if got != move["expect"]:
            diverged = {"move": i, "floor": move["floor"], "expected": move["expect"], "got": got}
            log(f"move {i} (F{move['floor']} {move['command']}) differs from the plan: "
                f"{ {k: (move['expect'].get(k), got.get(k)) for k in set(got) | set(move['expect']) if got.get(k) != move['expect'].get(k)} }")
            break
        if i % 200 == 0:
            log(f"move {i}/{len(route['moves'])}: F{got.get('floor')} hp {got.get('hp')}/{got.get('max_hp')}")
    if diverged and live.state.get("run") and not finish_with_policy:
        log("stopped: the live run no longer matches the plan; re-plan from its state, or rerun with "
            "--on-divergence policy to finish it with the plain trial policy")
    elif diverged and live.state.get("run"):
        log("finishing the run with the trial policy (no search live)")
        policy = TrialPolicy()
        while live.state.get("run"):
            decision = policy.decide(live.state)
            try:
                live.call(decision.command, decision.params)
            except GameError as err:
                if err.code != "invalid":
                    raise
                policy.ban(live.state["run"]["floor"], decision, message=str(err))
    departure = live.state.get("last_departure") or {}
    log(f"trial over: {departure}")
    return {"diverged": diverged, "departure": departure}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="search the deepest route in the local engine")
    plan.add_argument("--engine", required=True, help="path to a checkout of SlimeTROH/longniang-qqbot")
    plan.add_argument("--out", default=os.path.join(HERE, "logs", "trial"))
    play_live = sub.add_parser("replay", help="play a planned route on the live server")
    play_live.add_argument("--route", required=True)
    play_live.add_argument("--delay", type=float, default=0.05, help="seconds between commands")
    play_live.add_argument("--on-divergence", choices=("stop", "policy"), default="stop",
                           help="when the live run stops matching the plan: stop (default), or finish with the policy")
    play_live.add_argument("--i-understand-live", action="store_true",
                           help="required: the replay plays a real trial attempt on the account")
    args = ap.parse_args()
    if args.action == "replay":
        if not args.i_understand_live:
            raise SystemExit("pass --i-understand-live to play the route on the live server")
        from client import ElevatorClient
        client = ElevatorClient()
        client.login(os.environ["ELEVATOR_USER"], os.environ["ELEVATOR_PASSWORD"])
        client.refresh()
        with open(args.route, encoding="utf-8") as fh:
            route = json.load(fh)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(args.route), f"replay_{stamp}.jsonl")
        with open(path, "w", encoding="utf-8", buffering=1) as trace:
            replay(LiveTrial(client, args.delay, print), route, trace=trace,
                   finish_with_policy=args.on_divergence == "policy")
        return
    os.makedirs(args.out, exist_ok=True)
    engine = TrialEngine(args.engine, os.path.join(args.out, "engine.sqlite3"))
    engine.start()
    t0 = time.time()
    planner = Planner(engine)
    planner.run()
    print(f"planned: F{planner.last_floor}, {len(planner.route)} moves, {time.time() - t0:.0f}s")
    with open(os.path.join(args.out, "route.json"), "w", encoding="utf-8") as fh:
        json.dump({"week": engine.state.get("prophecy", {}).get("week"), "moves": planner.route}, fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
