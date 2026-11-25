"""Boss farming runner for the Endless Elevator.

Examples:
    python farm.py --mode sim --runs 3                     # free practice runs (no rewards)
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
from combat import Calibration
from policy import BossFarmPolicy, PolicyConfig

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
DEFAULT_LOADOUT = ("adrenaline=2,ice_tea=50,coffee=50,triage_vial=50,slow_serum=50,barrier_serum=50,"
                   "critical_serum=50,field_surgery=50")  # healing + adrenaline only; doll/pouch are kept at home
TICKET_PRICE = 350


def parse_loadout(text: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, qty = part.partition("=")
        out[key.strip()] = int(qty or 1)
    return out


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
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = ElevatorClient()
        self.cal = Calibration.load(os.path.join(HERE, "calibration.json"))
        self.policy = BossFarmPolicy(self.cal, PolicyConfig(max_floor=args.max_floor,
                                                             balance_gear_score=args.balance_gear_score))
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace = open(os.path.join(LOG_DIR, f"trace_{stamp}_{args.mode}.jsonl"), "a", encoding="utf-8")

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
        player = self.client.state["player"]
        if player.get("rush_floors") != 10 or not player.get("rush_skip_valuable"):
            self.call("rush_settings", {"floors": 10, "skip_valuable": True})

    # ------------------------------------------------------------- run loop
    def start_run(self, loadout: dict[str, int]) -> None:
        if self.args.mode == "sim":
            self.call("simulate", {"map_id": self.args.map})
            return
        owned = self.owned_supplies()
        supplies = [{"id": i, "quantity": min(q, owned[i])} for i, q in loadout.items() if owned[i] > 0]
        try:
            self.call("start", {"map_id": self.args.map, "supplies": supplies})
        except GameError as err:
            # Carry limits are undocumented: fall back to halving quantities until accepted.
            self.log(f"start rejected with supplies ({err}); retrying with smaller loadout")
            while supplies:
                supplies = [{"id": s["id"], "quantity": s["quantity"] // 2} for s in supplies if s["quantity"] > 1]
                try:
                    self.call("start", {"map_id": self.args.map, "supplies": supplies})
                    return
                except GameError:
                    continue
            self.call("start", {"map_id": self.args.map})

    def reset_tally(self) -> None:
        self.tally = {"t0": time.time(), "steps": 0, "bosses": 0, "rainbows": [], "floor": 0,
                      "outcome": "", "summary": ""}

    def play_run(self) -> dict:
        """Play until the run ends. Counters live in self.tally so they survive resumes."""
        t = self.tally
        while self.client.state.get("run"):
            state = self.client.state
            run = state["run"]
            floor = t["floor"] = run["floor"]
            decision = self.policy.decide(state)
            enemy = (run.get("event") or {}).get("enemy") if decision.command == "fight" else None
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
                                         "decision": decision.command, "params": decision.params,
                                         "reason": decision.reason, "result": result}, ensure_ascii=False) + "\n")
            if self.args.verbose or decision.command not in ("rush", "fight", "field_pick") or (enemy and enemy.get("boss")):
                self.log(f"F{floor} hp {run['hp']}/{run['max_hp']} -> {decision.command}: {decision.reason}"
                         f" | {result.get('summary', '')}")
            t["summary"], t["outcome"] = result.get("summary", t["summary"]), result.get("outcome", t["outcome"])
        self.cal.save()
        self.trace.flush()
        return {"floor": t["floor"], "bosses": t["bosses"], "rainbows": list(t["rainbows"]), "steps": t["steps"],
                "seconds": round(time.time() - t["t0"]), "outcome": t["outcome"], "summary": t["summary"]}

    @staticmethod
    def _loot(result: dict) -> list[dict]:
        items = list((result.get("battle") or {}).get("loot") or [])
        items += (result.get("pickups") or {}).get("loot") or []
        for fl in (result.get("rush") or {}).get("floors") or []:
            items += fl.get("loot") or []
        return items

    def record(self, summary: dict) -> None:
        path = os.path.join(LOG_DIR, "runs.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(["time", "mode", "map", "floor", "bosses", "rainbows", "steps", "seconds", "outcome", "summary"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), self.args.mode, self.args.map,
                             summary["floor"], summary["bosses"], "|".join(summary["rainbows"]),
                             summary["steps"], summary["seconds"], summary["outcome"], summary["summary"]])

    # ------------------------------------------------------------- main
    def run(self) -> None:
        self.client.login()
        self.client.refresh()
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
            self.client.refresh()
            if not self.client.state.get("run"):
                if self.args.mode == "real":
                    if all(self.rainbow_status().values()):
                        self.log("all rainbow items collected - stopping")
                        break
                    if not self.ensure_ticket():
                        self.log("no tickets available - stopping")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("sim", "real"), default="sim")
    ap.add_argument("--map", default="mall", help="map id (mall is optimal for boss farming)")
    ap.add_argument("--runs", type=int, default=1)
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
    args = ap.parse_args()
    if args.sell_now and args.mode != "real":
        sys.exit("--sell-now needs --mode real --i-understand-real")
    if args.mode == "real" and not args.i_understand_real and not args.status:
        sys.exit("Real mode spends tickets and gold. Re-run with --i-understand-real to confirm.")
    Farmer(args).run()


if __name__ == "__main__":
    main()
