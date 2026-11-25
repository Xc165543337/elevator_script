"""Boss farming runner for the Endless Elevator.

Examples:
    python farm.py --mode sim --runs 3                     # free practice runs (no rewards)
    python farm.py --mode sim --runs 3 --policy markov     # decisions from the solved MDP (mdp.py)
    python farm.py --mode real --runs 5 --i-understand-real # spends tickets/gold
    python farm.py --status                                 # show collection progress

Real mode stops automatically once all four rainbow items are unlocked.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import requests

from client import ElevatorClient, GameError
import heirlooms
from combat import Calibration, battle_enemy
from markov_policy import MarkovConfig, MarkovPolicy
from model import LEGACY_RULES, LIVE_RULES, floor_mix, parse_floor_odds
from policy import SUPPLY_HEAL, BossFarmPolicy, PolicyConfig

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
MAX_CARRY = 50   # live server limit per supply in one start request
# Every shop heal and smokes at the carry limit, plus the two adrenaline shots the policy uses on F1.
# Craft-only supplies (grenade, shield) would spend materials, and doll / pouch are kept at home.
DEFAULT_LOADOUT = ",".join(["adrenaline=2", *(f"{item}={MAX_CARRY}" for item in (*SUPPLY_HEAL, "smoke"))])
TICKET_PRICE = 350
# rush_settings parameter -> (state field, wanted value, set even if the server does not report it)
RUSH_SETTINGS = {
    "floors": ("rush_floors", 10, True),
    "skip_valuable": ("rush_skip_valuable", True, True),
    "event_mode": ("rush_event_mode", "pause", False),
    "auto_elite": ("rush_auto_elite", False, False),
}


def parse_loadout(text: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, qty = part.partition("=")
        out[key.strip()] = int(qty or 1)
    return out


def format_mix(mix: dict[str, float]) -> str:
    return " ".join(f"{kind} {p:.0%}" for kind, p in mix.items())


SELL_KINDS = ("collection", "material")   # supplies (potions, in-game items), crates and gear are never sold


def select_loot_to_sell(stock: list[dict], baseline: set[int], dedupe_rarities: set[str],
                        bulk_rarities: set[str]) -> list[tuple[dict, int]]:
    """(entry, quantity) pairs to sell.

    dedupe_rarities (red/gold): across the whole warehouse keep exactly one unit of each item id, the
        highest-valued one, and sell every other unit (old duplicates included).
    bulk_rarities (purple/green/white): sell everything that is new since `baseline` (this run's loot).
    """
    picked: list[tuple[dict, int]] = []
    sellable = [it for it in stock if it.get("kind") in SELL_KINDS and it.get("stock_id") is not None]
    groups: dict[str, list[dict]] = {}
    for it in sellable:
        if it.get("rarity") in dedupe_rarities:
            groups.setdefault(it["id"], []).append(it)
        elif it.get("rarity") in bulk_rarities and it["stock_id"] not in baseline:
            picked.append((it, it.get("quantity") or 1))
    for entries in groups.values():
        entries.sort(key=lambda x: (x.get("value") or 0, x["stock_id"]), reverse=True)
        keep, *others = entries
        if (keep.get("quantity") or 1) > 1:
            picked.append((keep, keep["quantity"] - 1))
        picked += [(it, it.get("quantity") or 1) for it in others]
    return picked


class Farmer:
    def __init__(self, args: argparse.Namespace, client=None):
        """`client` defaults to the live server; offline.py passes an in-process game engine."""
        self.args = args
        self.client = client or ElevatorClient()
        self.cal = Calibration.load(os.path.join(HERE, "calibration.json"))
        config = PolicyConfig(max_floor=args.max_floor, balance_gear_score=args.balance_gear_score,
                              relic_loot_value=args.relic_loot_value,
                              rules=LEGACY_RULES if args.rules == "legacy" else LIVE_RULES)
        if args.policy == "markov":
            self.policy = MarkovPolicy(self.cal, config, MarkovConfig(cache_dir=os.path.join(HERE, "cache")),
                                       log=self.log)
        else:
            self.policy = BossFarmPolicy(self.cal, config)
        self.focus: list[str] | None = None   # blessing sets pursued by the current real run
        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace = open(os.path.join(self.log_dir, f"trace_{stamp}_{args.mode}.jsonl"), "a", encoding="utf-8")

    # ------------------------------------------------------------- utilities
    def log(self, msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    def call(self, command: str, params: dict | None = None) -> dict:
        """Send a command with polite pacing and recovery from stale revisions / network errors."""
        attempt = cooldowns = 0
        while attempt < 5:
            try:
                time.sleep(self.args.delay)
                return self.client.command(command, params or {})
            except GameError as err:
                if err.code == "cooldown" and cooldowns < 30:  # server rate limit ("操作太快")
                    cooldowns += 1
                    time.sleep(1.0)
                    continue
                attempt += 1
                if err.code == "maintenance":
                    self.log("server maintenance; waiting 60s")
                    time.sleep(60)
                elif attempt < 3 and err.code not in (None, "invalid", "forbidden"):
                    self.log(f"retry after {err} ")
                else:
                    raise
                self.client.refresh()
            except (requests.RequestException, ValueError) as err:
                attempt += 1
                self.log(f"network error ({err}); backoff")
                time.sleep(2 ** attempt)
                self.safe_refresh()
        raise RuntimeError(f"command {command} failed repeatedly")

    def safe_refresh(self) -> bool:
        try:
            self.client.refresh()
            return True
        except (requests.RequestException, ValueError, GameError) as err:
            self.log(f"refresh failed ({err})")
            return False

    def connect(self, attempts: int = 30, wait: float = 60.0) -> None:
        """Log in and load the state, waiting out a server that is restarting (HTTP 502 and the like).
        Game errors such as wrong credentials are raised at once."""
        for attempt in range(1, attempts + 1):
            try:
                self.client.login()
                self.client.refresh()
                return
            except (requests.RequestException, ValueError) as err:
                if attempt == attempts:
                    raise
                self.log(f"server unavailable ({err}); retrying in {wait:.0f}s")
                time.sleep(wait)

    def play_run_supervised(self, retries: int = 10) -> dict:
        """play_run, resuming the same run after transient server/network failures."""
        for attempt in range(retries + 1):
            try:
                return self.play_run()
            except (requests.RequestException, RuntimeError) as err:
                if attempt == retries:
                    raise
                self.log(f"run interrupted ({err}); resuming in 30s")
                time.sleep(30)
                while not self.safe_refresh():
                    time.sleep(30)
        raise AssertionError("unreachable")

    def rainbow_status(self) -> dict[str, bool]:
        return {r["name"]: r["unlocked"] for r in self.client.state["atlas"]["rainbow"]}

    def print_status(self) -> None:
        st = self.client.state
        atlas, player = st["atlas"], st["player"]
        self.log(f"collection {atlas['known']}/{atlas['total']} | gold {player['gold']:,} | "
                 f"tickets {player['tickets']} (+{player['paid_tickets']} paid)")
        for name, unlocked in self.rainbow_status().items():
            self.log(f"  rainbow {name}: {'YES' if unlocked else 'no'}")
        missing = [i["name"] for i in atlas["items"] if not i["unlocked"] and i.get("rarity") != "rainbow"]
        if missing:
            self.log(f"  other missing: {', '.join(missing)}")
        for line in heirlooms.collection_lines(st.get("heirlooms")):
            self.log(f"  blessing set {line}")

    # ------------------------------------------------------------- preparation
    def owned_supplies(self) -> Counter:
        stock = Counter()
        for item in self.client.state.get("supplies") or []:
            stock[item["id"]] += item.get("quantity", 1)
        return stock

    def ensure_ticket(self) -> bool:
        player = self.client.state["player"]
        if player["tickets"] + player["paid_tickets"] > 0:
            return True
        if not self.args.buy_tickets or player["gold"] - TICKET_PRICE < self.args.gold_reserve:
            return False
        try:
            self.call("buy", {"id": "ticket"})
        except GameError as err:
            self.log(f"cannot buy ticket: {err}")
            return False
        player = self.client.state["player"]
        return player["tickets"] + player["paid_tickets"] > 0

    def restock(self, loadout: dict[str, int]) -> None:
        prices = {it["id"]: it["price"] for it in self.client.state.get("shop") or []}
        owned = self.owned_supplies()
        for item_id, want in loadout.items():
            missing = want - owned[item_id]
            if missing <= 0:
                continue
            if item_id not in prices:
                self.craft(item_id, missing)
                continue
            affordable = int((self.client.state["player"]["gold"] - self.args.gold_reserve) // prices[item_id])
            qty = min(missing, max(0, affordable))
            if qty <= 0:
                self.log("gold reserve reached; stop restocking")
                return
            try:
                self.call("buy", {"id": item_id, "quantity": qty})  # one call for the whole amount
            except GameError as err:
                self.log(f"bulk buy rejected ({err}); buying one at a time")
                for _ in range(qty):
                    self.call("buy", {"id": item_id})
            self.log(f"bought {qty} x {item_id}")

    def craft(self, item_id: str, count: int) -> None:
        """Black-market items (e.g. pouch, grenade): craft while materials allow."""
        made = 0
        for _ in range(count):
            recipe = next((m for m in self.client.state.get("market") or [] if m["id"] == item_id), None)
            if not recipe or not recipe.get("ready"):
                break
            self.call("craft", {"id": item_id})
            made += 1
        if made:
            self.log(f"crafted {made} x {item_id}")
        if made < count:
            self.log(f"short of {item_id}: {count - made} missing (not in shop, materials insufficient)")

    def warehouse_ids(self) -> set[int]:
        return {it["stock_id"] for it in self.client.state.get("warehouse") or [] if it.get("stock_id") is not None}

    def sell_run_loot(self, baseline: set[int]) -> None:
        """Apply the selling rule (see select_loot_to_sell) after a finished run."""
        self.safe_refresh()
        self.log("selling loot (one call per item; this can take several minutes)")
        earned = sold = 0
        stock = self.client.state.get("warehouse") or []
        for it, qty in select_loot_to_sell(stock, baseline, set(self.args.dedupe_rarities.split(",")),
                                           set(self.args.sell_rarities.split(","))):
            try:
                self.call("sell", {"stock_id": it["stock_id"], "quantity": qty})
            except GameError as err:
                self.log(f"could not sell {it.get('name')} ({err})")
                continue
            sold += qty
            earned += (it.get("value") or 0) * qty
        self.log(f"sold {sold} run items (~{earned:,} gold)")

    def ensure_rush_settings(self) -> None:
        """Rush the way the models assume: 10 floors, no pause for gold / red loot, and a stop at events
        and before elites, so each of those is decided individually. Settings a server does not report
        (older versions) are left alone."""
        player = self.client.state["player"]
        change = {key: want for key, (field, want, required) in RUSH_SETTINGS.items()
                  if (required or field in player) and player.get(field) != want}
        if not change:
            return
        try:
            self.call("rush_settings", change)
            self.log(f"rush settings changed: {change}")
        except GameError as err:   # e.g. a server that refuses changes during a run
            self.log(f"could not change rush settings {change}: {err}")

    # ------------------------------------------------------------- run loop
    def pursuit(self) -> list[str] | None:
        """Blessing sets to pursue on a real run (None: the server has no blessings). Raises FocusError."""
        return heirlooms.resolve_focus(self.client.state.get("heirlooms"), self.args.focus)

    def start_run(self, loadout: dict[str, int]) -> None:
        if self.args.mode == "sim":
            self.call("simulate", {"map_id": self.args.map})
            return
        base = {"map_id": self.args.map, **({"focus": self.focus} if self.focus else {})}
        owned = self.owned_supplies()
        supplies = [{"id": i, "quantity": min(q, owned[i])} for i, q in loadout.items() if owned[i] > 0]
        try:
            self.call("start", {**base, "supplies": supplies})
        except GameError as err:
            # Carry limits are undocumented: fall back to halving quantities until accepted.
            self.log(f"start rejected with supplies ({err}); retrying with smaller loadout")
            while supplies:
                supplies = [{"id": s["id"], "quantity": s["quantity"] // 2} for s in supplies if s["quantity"] > 1]
                try:
                    self.call("start", {**base, "supplies": supplies})
                    return
                except GameError:
                    continue
            self.call("start", base)

    def ensure_pursuit(self) -> None:
        """A running real run that still lacks its blessing choice (本局补选追寻) gets it now."""
        if (self.client.state.get("heirlooms") or {}).get("needs_focus") and self.focus:
            self.call("pursue", {"focus": self.focus})
            self.log(f"pursuing blessing sets {', '.join(self.focus)}")

    def reset_tally(self) -> None:
        self.tally = {"t0": time.time(), "steps": 0, "bosses": 0, "rainbows": [], "floor": 0,
                      "outcome": "", "summary": "", "heirlooms": heirlooms.owned(self.client.state.get("heirlooms")),
                      "odds_checked": False}

    def check_floor_odds(self, run: dict) -> None:
        """Once per run, compare the encounter odds the client shows for the next floor with the model's
        (they changed with the server update of 2026-11-25) and warn on a mismatch."""
        floor = run["floor"] + 1
        odds = parse_floor_odds(run.get("next_floor"))
        if self.tally["odds_checked"] or odds is None or floor < 60:   # the blend from F1 has settled by F60
            return
        self.tally["odds_checked"] = True
        expected = floor_mix(floor, self.policy.cfg.rules)
        if max(abs(odds[k] - expected[k]) for k in odds) > 0.015:
            self.log(f"WARNING: encounter odds at F{floor} are {format_mix(odds)} on the server but "
                     f"{format_mix(expected)} in the model; update model.ServerRules.deep_mix")
        else:
            self.log(f"encounter odds at F{floor} match the model ({format_mix(odds)})")

    def play_run(self) -> dict:
        """Play until the run ends. Counters live in self.tally so they survive resumes."""
        t = self.tally
        preview = None   # a wish preview comes back in the command result, not in the state
        while self.client.state.get("run"):
            self.ensure_pursuit()
            state = self.client.state
            if preview:
                state = {**state, "wish_preview": preview}
            run = state["run"]
            self.check_floor_odds(run)
            floor = t["floor"] = run["floor"]
            decision = self.policy.decide(state)
            killing = decision.command in ("fight", "wish_confirm")   # a confirmed wish kills the enemy outright
            enemy = (run.get("event") or {}).get("enemy") if killing else None
            enemy = battle_enemy(state, enemy) if enemy else None
            fighter = self.policy.current_build(state) if enemy else None
            try:
                data = self.call(decision.command, decision.params)
            except GameError as err:
                if err.code != "invalid":
                    raise
                self.log(f"F{floor} rejected {decision.command} {decision.params}: {err}; choosing again")
                used = next((it.get("id") for it in run.get("bag") or []
                             if decision.command == "use" and it.get("index") == decision.params.get("index")), None)
                self.policy.ban(floor, decision, item_id=used, message=str(err))
                self.client.refresh()
                continue
            t["steps"] += 1
            result = data.get("result") or {}
            preview = result.get("wish_preview")
            self.policy.note(result.get("limit_notices") or [])
            battle = result.get("battle")
            if enemy and battle and battle.get("log"):
                self.cal.observe(fighter, enemy, battle["log"])
                if battle.get("outcome") == "victory" and enemy.get("boss"):
                    t["bosses"] += 1
                    self.log(f"BOSS #{t['bosses']} down at F{floor}: {enemy.get('name')}")
            for item in self._loot(result):
                if item.get("rarity") == "rainbow" or str(item.get("id", "")).startswith("prism_"):
                    t["rainbows"].append(item.get("name"))
                    self.log(f"*** RAINBOW DROP: {item.get('name')} ***")
            self.trace.write(json.dumps({"t": time.time(), "floor": floor, "hp": run["hp"],
                                         "stats": run.get("effective_stats"), "dmg_mult": run.get("damage_multiplier"),
                                         "slots": run.get("field_slots"), "enemy": (run.get("event") or {}).get("enemy"),
                                         "condition": (state["player"].get("secondary") or {}).get("condition"),
                                         "decision": decision.command, "params": decision.params,
                                         "reason": decision.reason, "result": result}, ensure_ascii=False) + "\n")
            if self.args.verbose or decision.command not in ("rush", "fight", "field_pick") or (enemy and enemy.get("boss")):
                self.log(f"F{floor} hp {run['hp']}/{run['max_hp']} -> {decision.command}: {decision.reason}"
                         f" | {result.get('summary', '')}")
            t["summary"], t["outcome"] = result.get("summary", t["summary"]), result.get("outcome", t["outcome"])
        self.cal.save()
        self.trace.flush()
        found = self.client.state.get("heirlooms")
        shining = heirlooms.new_pieces(t["heirlooms"], heirlooms.owned(found), found)
        for name in shining:
            self.log(f"*** SHINING DROP: {name} ***")
        return {"floor": t["floor"], "bosses": t["bosses"], "rainbows": list(t["rainbows"]), "shining": shining,
                "steps": t["steps"], "seconds": round(time.time() - t["t0"]), "outcome": t["outcome"],
                "summary": t["summary"]}

    @staticmethod
    def _loot(result: dict) -> list[dict]:
        items = list((result.get("battle") or {}).get("loot") or [])
        items += (result.get("pickups") or {}).get("loot") or []
        for fl in (result.get("rush") or {}).get("floors") or []:
            items += fl.get("loot") or []
        return items

    def record(self, summary: dict) -> None:
        path = os.path.join(self.log_dir, "runs.csv")
        row = {"time": datetime.now().isoformat(timespec="seconds"), "mode": self.args.mode,
               "policy": self.args.policy, "map": self.args.map, "floor": summary["floor"],
               "bosses": summary["bosses"], "rainbows": "|".join(summary["rainbows"]),
               "shining": "|".join(summary.get("shining") or []), "steps": summary["steps"],
               "seconds": summary["seconds"], "outcome": summary["outcome"], "summary": summary["summary"]}
        header = list(row)
        if os.path.exists(path):  # keep an existing file's columns (older files lack "policy" and "shining")
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), header)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------- main
    def run(self) -> None:
        self.connect()
        self.print_status()
        if self.args.status:
            return
        if self.args.sell_now:  # red/gold dedupe across the warehouse only (no run loot to sell)
            self.sell_run_loot(self.warehouse_ids())
            self.print_status()
            return
        loadout = parse_loadout(self.args.loadout)
        total_bosses = 0
        session_baseline = baseline = self.warehouse_ids()  # loot already at home is never bulk-sold
        for n in range(1, self.args.runs + 1):
            while not self.safe_refresh():   # between runs too, a restarting server is waited out
                time.sleep(30)
            if not self.client.state.get("run"):
                if self.args.mode == "real":
                    if all(self.rainbow_status().values()):
                        self.log("all rainbow items collected - stopping")
                        break
                    if not self.ensure_ticket():
                        self.log("no tickets available - stopping")
                        break
                    try:
                        self.focus = self.pursuit()
                    except heirlooms.FocusError as err:
                        self.log(f"cannot start a real run: {err}")
                        break
                    if self.args.restock:
                        self.restock(loadout)
                self.ensure_rush_settings()
                self.policy.new_run()
                self.reset_tally()
                self.client.refresh()
                baseline = self.warehouse_ids()
                self.start_run(loadout)
                self.log(f"run {n} started ({self.args.mode}, {self.args.map})")
            else:
                is_sim = bool(self.client.state["run"].get("simulation"))
                if is_sim != (self.args.mode == "sim"):
                    self.log(f"an active {'practice' if is_sim else 'REAL'} run exists; refusing to touch it in "
                             f"{self.args.mode} mode. Finish it with the matching --mode.")
                    break
                if not is_sim:
                    try:
                        self.focus = self.pursuit()
                    except heirlooms.FocusError as err:
                        self.focus = None   # only needed if the server asks for it (logged when it does)
                        if (self.client.state.get("heirlooms") or {}).get("needs_focus"):
                            self.log(f"this run still needs its blessing choice: {err}")
                            break
                self.ensure_rush_settings()
                self.reset_tally()
                self.log(f"resuming run at F{self.client.state['run']['floor']}")
            summary = self.play_run_supervised()
            total_bosses += summary["bosses"]
            self.record(summary)
            self.log(f"run {n} done: F{summary['floor']} bosses {summary['bosses']} steps {summary['steps']} "
                     f"{summary['seconds']}s | {summary['summary']} | total bosses {total_bosses}")
            if self.args.mode == "real" and self.args.sell_loot == "each":
                self.sell_run_loot(baseline)
        if self.args.mode == "real" and self.args.sell_loot == "end":
            self.log("selling the loot of this session")
            self.sell_run_loot(session_baseline)
        self.client.refresh()
        self.print_status()
        self.log(f"calibration: k_player={self.cal.k_player:.3f} k_enemy={self.cal.k_enemy:.3f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("sim", "real"), default="sim")
    ap.add_argument("--policy", choices=("threshold", "markov"), default="threshold",
                    help="threshold = hand-tuned rules (policy.py); markov = solved MDP (markov_policy.py)")
    ap.add_argument("--log-dir", default=LOG_DIR, help="where traces and runs.csv are written")
    ap.add_argument("--map", default="mall", help="map id (mall is optimal for boss farming)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--focus", default=None,
                    help="real runs: the two blessing sets to pursue (追寻), by number or id, e.g. 1,8; "
                         "default: the pair chosen last time")
    ap.add_argument("--relic-loot-value", type=int, default=PolicyConfig.relic_loot_value,
                    help="loot (舔包) fallen players' bags worth at least this much gold instead of returning them "
                         "for the bounty; loot that valuable which the safe box cannot hold is banked by extracting")
    ap.add_argument("--rules", choices=("live", "legacy"), default="live",
                    help="server rules: live = elites pause after round 8 of 16 and the current encounter odds "
                         "(server since 2026-11-25); legacy = the older engine of the public bot repository (offline.py)")
    ap.add_argument("--loadout", default=DEFAULT_LOADOUT, help="supplies to carry, e.g. field_surgery=12,coffee=10")
    ap.add_argument("--no-restock", dest="restock", action="store_false", help="do not buy missing supplies")
    ap.add_argument("--buy-tickets", action="store_true", help="buy paid tickets (350 gold) when free ones run out")
    ap.add_argument("--gold-reserve", type=int, default=1_000_000, help="never spend below this gold balance")
    ap.add_argument("--max-floor", type=int, default=None, help="extract on reaching this floor")
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between API calls")
    ap.add_argument("--balance-gear-score", type=float, default=-1.0,
                    help="wear score for the x2/x2 temp gear (negative = never wear it)")
    ap.add_argument("--sell-loot", choices=("off", "end", "each"), default="off",
                    help="loot selling (one server call per item, slow): off = never; end = once after the "
                         "last run of this session; each = after every run")
    ap.add_argument("--sell-rarities", default="purple,green,white",
                    help="rarities whose new run loot is sold entirely (blue is kept for rerolls)")
    ap.add_argument("--dedupe-rarities", default="red,gold",
                    help="rarities reduced to one unit (the highest-valued) per item across the warehouse")
    ap.add_argument("--sell-now", action="store_true",
                    help="apply the red/gold one-copy rule to the warehouse now, then exit (requires --mode real)")
    ap.add_argument("--status", action="store_true", help="print collection status and exit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--i-understand-real", action="store_true", help="required for --mode real")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.sell_now and args.mode != "real":
        sys.exit("--sell-now needs --mode real --i-understand-real")
    if args.mode == "real" and not args.i_understand_real and not args.status:
        sys.exit("Real mode spends tickets and gold. Re-run with --i-understand-real to confirm.")
    Farmer(args).run()


if __name__ == "__main__":
    main()
