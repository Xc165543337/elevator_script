"""Boss-farming decision policy.

Goal: kill as many bosses as possible per ticket. Rainbow drops go straight to the
warehouse, so the run's bag is secondary; depth (= boss count) is what matters.

The policy is a pure function of the current game state: `decide(state) -> Decision`.
It never mutates the state and is safe to unit-test against recorded states.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from combat import Calibration, FightEstimate, chain_risk, estimate, run_build
from gear import GearModel
from model import Build

# Heal amount (fraction of max HP) of shop supplies; field items are parsed from text.
SUPPLY_HEAL = {
    "coffee": 0.40, "ice_tea": 0.25, "triage_vial": 0.57, "barrier_serum": 0.50,
    "slow_serum": 0.38, "critical_serum": 0.55, "field_surgery": 1.00,
}
COMBAT_ONLY = {"field_surgery"}
MAX_DURATION_BUFFS = 3  # server allows at most three floor-limited buffs at once
# Supplies kept in the bag (usable during the run), never moved into the safe box.
KEEP_IN_BAG = set(SUPPLY_HEAL) | {"adrenaline", "grenade", "smoke", "shield", "flare", "pouch", "doll"}
SUPPLY_COST = {
    "coffee": 300, "ice_tea": 150, "triage_vial": 780, "barrier_serum": 1450,
    "slow_serum": 1250, "critical_serum": 2300, "field_surgery": 3300,
}
# Field-item effect keywords that help in a fight (used before risky fights).
COMBAT_BUFF_WORDS = ("攻击+", "防御+", "减伤", "闪避率+", "吸血", "首回合", "对首领", "暴击率+", "穿透+", "伤害 ×", "护盾")
# Weights to score wearable field gear from its description.
WEAR_WEIGHTS = {"攻击": 1.0, "防御": 0.7, "减伤": 1.3, "闪避率": 1.5, "吸血": 1.4, "对首领伤害": 0.8,
                "暴击率": 0.6, "穿透": 0.6, "追击率": 0.4, "敏捷": 0.25, "首回合攻击": 0.2, "逃跑率": 0.0}


@dataclass
class PolicyConfig:
    max_death_risk: float = 0.02        # accepted P(death) for a regular fight
    max_boss_window_risk: float = 0.03  # accepted P(death) within one boss window
    abort_risk: float = 0.10            # above this with no resources left: extract instead of fighting
    extract_when_dry: bool = True       # no healing left at all -> extract at the next safe moment
    greedy_with_doll: bool = True       # a 替身娃娃 in the bag saves run+bag on death: keep pushing for bosses
    rush_min_hp: float = 0.55           # below this HP fraction, heal or step floor-by-floor
    max_floor: int | None = None        # optional hard stop (extract on reaching it)
    balance_gear_score: float = -1.0    # score for 策划的平衡性调整 (x2 dealt / x2 taken); <0 = never wear
    samples: int = 600
    start_adrenaline: int = 2           # 暴走肾上腺素 shots to use on F1 (max 2 stacks)
    max_rush_risk: float = 0.01         # max P(death) over 4 back-to-back normal fights to allow a rush
    swap_margin: float = 1.2            # swap loot into a full safe box only if worth >= margin x cheapest secured
    use_pouch: bool = False             # 哆啦的临时扩容袋 is precious: never use it (found ones go home)
    combat_topup_hp: float = 0.30       # with only combat-only heals left, use one in an encounter below this HP


@dataclass
class Decision:
    command: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    estimate: FightEstimate | None = None


def _pct(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) / 100 if m else 0.0


def _controls(state: dict) -> dict[str, dict]:
    return {c["command"]: c for c in state.get("controls") or []}


class BossFarmPolicy:
    def __init__(self, cal: Calibration, config: PolicyConfig | None = None):
        self.cal = cal
        self.cfg = config or PolicyConfig()
        # (floor, command, params) tuples the server rejected; never proposed again on that floor.
        self.banned: set[tuple] = set()
        self.event_atk_capped = False   # server reported the +60% event attack cap this run
        self.adrenaline_used = 0
        self.pouch_used = False
        self._base_stats: dict | None = None   # effective stats last seen with no floor-limited buffs
        self._affix_build: Build | None = None
        self._loadout: tuple | None = None

    def new_run(self) -> None:
        self.banned.clear()
        self.event_atk_capped = False
        self.adrenaline_used = 0
        self.pouch_used = False
        self._base_stats = None

    def note(self, notices: list[str]) -> None:
        """React to server limit notices."""
        for text in notices:
            if "奇遇攻击加成已达" in text:
                self.event_atk_capped = True
            if "扩容袋" in text:
                self.pouch_used = True

    def ban(self, floor: int, decision: "Decision") -> None:
        self.banned.add((floor, decision.command, repr(sorted(decision.params.items()))))

    def _allowed(self, floor: int, command: str, params: dict) -> bool:
        return (floor, command, repr(sorted(params.items()))) not in self.banned

    # ------------------------------------------------------------------ helpers
    def current_build(self, state: dict) -> Build:
        """Live combat build; permanent affixes are derived once per equipped loadout."""
        loadout = tuple(sorted((state["player"].get("loadout") or {}).items()))
        if self._affix_build is None or loadout != self._loadout:
            gm = GearModel(state)
            self._affix_build, self._loadout = gm.build(gm.current), loadout
        return run_build(state, self._affix_build)

    def lasting_build(self, state: dict) -> Build:
        """Build without floor-limited buffs (they can expire during a multi-floor rush).
        Uses the effective stats last observed while no such buff was active."""
        run = state["run"]
        if not (run.get("field_buffs") or []):
            self._base_stats = dict(run["effective_stats"])
        build = self.current_build(state)
        base = self._base_stats
        if base:
            build = replace(build, atk=min(build.atk, base["atk"]), defense=min(build.defense, base["def"]),
                            agi=min(build.agi, base["agi"]))
        return build

    def estimate(self, state: dict) -> FightEstimate:
        return estimate(self.current_build(state), state, self.cal, samples=self.cfg.samples)

    @staticmethod
    def _field_heals(run: dict) -> list[tuple[float, dict]]:
        out = []
        for item in run.get("field") or []:
            if item.get("healing") and item.get("quantity", 0) > 0:
                out.append((_pct(item.get("description", ""), r"回复 (\d+)% 最大生命"), item))
        return out

    @staticmethod
    def _bag_heals(run: dict, in_combat: bool) -> list[tuple[float, dict]]:
        out = []
        for item in run.get("bag") or []:
            iid = item.get("id")
            if iid in SUPPLY_HEAL and (in_combat or iid not in COMBAT_ONLY):
                heal = SUPPLY_HEAL[iid]
                if iid == "critical_serum" and run["hp"] <= 0.35 * run["max_hp"]:
                    heal = 0.90
                out.append((heal, item))
        return out

    @staticmethod
    def _active_buff_ids(run: dict) -> set[str]:
        ids = set()
        for entry in (run.get("field_buffs") or []) + (run.get("field_charges") or []):
            ids.add(entry.get("id") if isinstance(entry, dict) else str(entry))
        return ids

    def _best_heal(self, run: dict, in_combat: bool, force: bool) -> Decision | None:
        """Pick the heal that wastes least. Free field items first, then cheapest supply per HP."""
        deficit = 1 - run["hp"] / run["max_hp"]
        if deficit <= 0.02:
            return None
        floor = run["floor"]
        field_heals = [(h, it) for h, it in self._field_heals(run)
                       if self._allowed(floor, "field_use", {"id": it["id"]})]
        fitting = [(h, it) for h, it in field_heals if h <= deficit + 0.05]
        if fitting or (force and field_heals):
            heal, item = max(fitting, key=lambda x: x[0]) if fitting else min(field_heals, key=lambda x: x[0])
            return Decision("field_use", {"id": item["id"]}, f"field heal {item['name']} (+{heal:.0%}, deficit {deficit:.0%})")
        bag = [(h, it) for h, it in self._bag_heals(run, in_combat)
               if self._allowed(floor, "use", {"index": it["index"]})]
        if not bag:
            return None
        fitting = [(h, it) for h, it in bag if h <= deficit + 0.10]
        if fitting:
            heal, item = max(fitting, key=lambda x: (x[0], -SUPPLY_COST.get(x[1]["id"], 0)))
        elif force:
            heal, item = min(bag, key=lambda x: SUPPLY_COST.get(x[1]["id"], 0))
        else:
            return None
        return Decision("use", {"index": item["index"]}, f"use {item['name']} (+{heal:.0%}, deficit {deficit:.0%})")

    def _combat_only_topup(self, run: dict) -> Decision | None:
        """Low HP and only combat-only heals (战地手术匣) left: use one now, during an encounter,
        so the run can go back to rushing instead of bleeding out over several easy fights."""
        if run["hp"] >= self.cfg.combat_topup_hp * run["max_hp"] or self._bag_heals(run, in_combat=False):
            return None
        floor = run["floor"]
        kits = [it for heal, it in self._bag_heals(run, in_combat=True)
                if it["id"] in COMBAT_ONLY and self._allowed(floor, "use", {"index": it["index"]})]
        if not kits:
            return None
        return Decision("use", {"index": kits[0]["index"]},
                        f"HP {run['hp'] / run['max_hp']:.0%} and only combat-only heals left -> {kits[0]['name']}")

    def _combat_buff(self, run: dict, boss: bool) -> Decision | None:
        active = self._active_buff_ids(run)
        duration_slots_full = len(run.get("field_buffs") or []) >= MAX_DURATION_BUFFS
        for item in run.get("field") or []:
            desc = item.get("description", "")
            if item.get("healing") or item.get("quantity", 0) <= 0 or item["id"] in active:
                continue
            if (duration_slots_full and "持续" in desc) or not self._allowed(run["floor"], "field_use", {"id": item["id"]}):
                continue
            if "对首领" in desc and not boss:
                continue
            if any(word in desc for word in COMBAT_BUFF_WORDS):
                return Decision("field_use", {"id": item["id"]}, f"buff {item['name']}: {desc}")
        return None

    @staticmethod
    def _bag_item(run: dict, item_id: str) -> dict | None:
        return next((it for it in run.get("bag") or [] if it.get("id") == item_id), None)

    # ------------------------------------------------------------------ decisions
    def decide(self, state: dict) -> Decision:
        """Best decision that the server has not already rejected on this floor."""
        decision = self._decide(state)
        if self._allowed(state["run"]["floor"], decision.command, decision.params):
            return decision
        ctr = _controls(state)
        enemy = (state["run"].get("event") or {}).get("enemy")
        for command in (("fight",) if enemy else ()) + ("rush", "down", "leave", "extract"):
            if command in ctr and self._allowed(state["run"]["floor"], command, {}):
                return Decision(command, {}, f"fallback after rejected {decision.command}")
        raise RuntimeError(f"all moves rejected on floor {state['run']['floor']}")

    def _decide(self, state: dict) -> Decision:
        run = state["run"]
        ctr = _controls(state)
        if run.get("field_offer"):
            return self._field_offer(run)
        boost = self._start_boost(run)
        if boost:
            return boost
        protect = self._protect_collectible(state, ctr) or self._secure_loot(state)
        if protect:
            return protect
        event = run.get("event") or {}
        if event.get("enemy") and ("fight" in ctr):
            return self._fight(state, ctr)
        if event.get("kind") == "choice" and any(c["command"] == "event" for c in state.get("controls") or []):
            return self._choice(state)
        if "rescue" in ctr:
            return Decision("rescue", ctr["rescue"].get("params", {}), "return fallen player's relic")
        if "down" in ctr or "rush" in ctr:
            return self._advance(state, ctr)
        if "leave" in ctr:
            return Decision("leave", {}, "nothing actionable")
        if "extract" in ctr:
            return Decision("extract", {}, "only extract available")
        raise RuntimeError(f"no known control: {list(ctr)} actions={state.get('actions')}")

    def _start_boost(self, run: dict) -> Decision | None:
        """Use carried adrenaline early (run-long +25% ATK each, max 2); leech/kill heals cover the HP cost.
        Also opens a carried safe-box pouch (+1 slot) right away."""
        if (run.get("event") or {}).get("enemy"):
            return None
        pouch = self._bag_item(run, "pouch") if self.cfg.use_pouch and not self.pouch_used else None
        if pouch and self._allowed(run["floor"], "use", {"index": pouch["index"]}):
            self.pouch_used = True  # one pouch per run; extra pouches stay in the bag
            return Decision("use", {"index": pouch["index"]}, "open safe-box pouch (+1 slot)")
        if self.adrenaline_used >= self.cfg.start_adrenaline:
            return None
        shot = self._bag_item(run, "adrenaline")
        if not shot or run["hp"] < 0.5 * run["max_hp"] or not self._allowed(run["floor"], "use", {"index": shot["index"]}):
            return None
        self.adrenaline_used += 1
        return Decision("use", {"index": shot["index"]}, f"adrenaline {self.adrenaline_used}/{self.cfg.start_adrenaline}")

    @staticmethod
    def missing_collectibles(state: dict) -> set[str]:
        atlas = state.get("atlas") or {}
        return {i["id"] for i in atlas.get("items") or [] if not i.get("unlocked") and i.get("rarity") != "rainbow"}

    def _protect_collectible(self, state: dict, ctr: dict) -> Decision | None:
        """Lock still-missing collection items into the safe box (kept even on death)."""
        run = state["run"]
        missing = self.missing_collectibles(state)
        secured = {it.get("id") for it in run.get("safe") or []}
        pending = [it for it in run.get("bag") or [] if it.get("id") in missing and it.get("id") not in secured]
        if not pending:
            return None
        item = pending[0]
        if item.get("can_protect") and len(run.get("safe") or []) < (run.get("safe_capacity") or 0):
            return Decision("save_box", {"index": item["index"]}, f"secure missing collectible {item['name']}")
        if "extract" in ctr and not (run.get("event") or {}).get("enemy"):
            return Decision("extract", {}, f"safe box full; extract to bank {item['name']}")
        return None

    def _secure_loot(self, state: dict) -> Decision | None:
        """Keep the most valuable loot in the safe box (kept even if the run fails)."""
        run = state["run"]
        if (run.get("event") or {}).get("enemy"):
            return None
        missing = self.missing_collectibles(state)
        floor = run["floor"]

        def worth(item: dict) -> float:
            if item.get("id") in missing:
                return float("inf")
            return float(item.get("value") or 0) * (item.get("quantity") or 1)

        candidates = [it for it in run.get("bag") or []
                      if it.get("can_protect") and it.get("id") not in KEEP_IN_BAG and worth(it) > 0]
        if not candidates:
            return None
        best = max(candidates, key=worth)
        safe = run.get("safe") or []
        if len(safe) < (run.get("safe_capacity") or 0):
            params = {"index": best["index"]}
            if self._allowed(floor, "save_box", params):
                return Decision("save_box", params, f"secure {best['name']} ({worth(best):,.0f})")
            return None
        if not safe:
            return None
        cheapest_pos, cheapest = min(enumerate(safe), key=lambda p: worth(p[1]))
        if worth(best) >= self.cfg.swap_margin * worth(cheapest) and worth(cheapest) != float("inf"):
            safe_index = int(cheapest.get("index") or cheapest.get("reference") or cheapest_pos + 1)
            params = {"index": best["index"], "safe_index": safe_index}
            if self._allowed(floor, "swap_box", params):
                return Decision("swap_box", params,
                                f"swap {best['name']} ({worth(best):,.0f}) for {cheapest.get('name')} ({worth(cheapest):,.0f})")
        return None

    def _fight(self, state: dict, ctr: dict) -> Decision:
        run = state["run"]
        enemy = run["event"]["enemy"]
        boss = bool(enemy.get("boss"))
        est = self.estimate(state)
        limit = self.cfg.max_boss_window_risk if boss else self.cfg.max_death_risk
        tag = f"{'boss' if boss else 'elite' if enemy.get('elite') else 'mob'} {enemy.get('name')}"
        topup = self._combat_only_topup(run)
        if topup:
            topup.reason = f"{tag}: {topup.reason}"
            return topup
        if est.death <= limit:
            return Decision("fight", {}, f"{tag}: death {est.death:.1%} win {est.win:.0%}", est)
        # Spend resources, one per step; the loop re-evaluates after each.
        for option in (lambda: self._best_heal(run, in_combat=True, force=False),
                       lambda: self._combat_buff(run, boss),
                       lambda: self._best_heal(run, in_combat=True, force=True)):
            decision = option()
            if decision:
                decision.reason = f"{tag}: death {est.death:.1%} -> {decision.reason}"
                decision.estimate = est
                return decision
        grenade = self._bag_item(run, "grenade")
        if grenade and est.death > limit:
            return Decision("use", {"index": grenade["index"]}, f"{tag}: grenade (death {est.death:.1%})", est)
        dry = self.cfg.extract_when_dry and not self.has_healing(run)
        if not self.greedy(run) and (est.death > self.cfg.abort_risk or (dry and est.death > limit)):
            if "extract" in ctr:
                return Decision("extract", {}, f"{tag}: death {est.death:.1%}, no resources -> extract", est)
            smoke = self._bag_item(run, "smoke")
            if smoke and not boss:
                return Decision("use", {"index": smoke["index"]}, f"{tag}: smoke out (death {est.death:.1%})", est)
            if "flee" in ctr:
                return Decision("flee", {}, f"{tag}: flee ({ctr['flee']['label']}), death {est.death:.1%}", est)
        return Decision("fight", {}, f"{tag}: forced fight, death {est.death:.1%}", est)

    def _choice(self, state: dict) -> Decision:
        run = state["run"]
        hp_frac = run["hp"] / run["max_hp"]
        best, best_score = None, float("-inf")
        for c in state.get("controls") or []:
            if c["command"] != "event" or not self._allowed(run["floor"], "event", c["params"]):
                continue
            score = self._score_option(c.get("description", ""), hp_frac, run["max_hp"], self.event_atk_capped)
            if score > best_score:
                best, best_score = c, score
        if best is None or best_score < 0:
            return Decision("leave", {}, f"event {run['event'].get('name')}: no worthwhile option")
        return Decision("event", best["params"], f"event {run['event'].get('name')}: {best['label']} ({best_score:.0f})")

    @staticmethod
    def _score_option(desc: str, hp_frac: float, max_hp: float, atk_capped: bool = False) -> float:
        """Score in HP-equivalents. Unknown upside counts as a small positive."""
        if "无消耗" in desc:
            return 0.0
        m = re.search(r"(\d+)%(?:成功|→)", desc)
        p = float(m.group(1)) / 100 if m else 1.0
        success = desc.split("失败")[0]
        failure = desc.split("失败")[1] if "失败" in desc else ""
        gain = 5.0  # loot / misc
        heal = _pct(success, r"回(\d+)%血")
        gain += heal * max_hp * min(1.0, (1 - hp_frac) * 2)
        if not atk_capped:
            gain += _pct(success, r"攻\+(\d+)%") * 800  # run-long attack buff (event cap +60%)
        gain += _pct(success, r"防\+(\d+)%") * 500
        if "偷看" in success or "侦察" in success:
            gain += 2
        fail_cost = float(m2.group(1)) if (m2 := re.search(r"扣约(\d+)HP", failure)) else 0.0
        cost = _pct(desc, r"耗 (\d+)% 当前血") * hp_frac * max_hp
        if (m3 := re.search(r"耗.*?约(\d+)", desc)) and not cost:
            cost = float(m3.group(1))
        if hp_frac < 0.5:
            cost *= 2
            fail_cost *= 2
        return p * gain - (1 - p) * fail_cost * 1.5 - cost

    def _score_wear(self, desc: str) -> float:
        if "伤害 ×2" in desc:
            return self.cfg.balance_gear_score
        return self._wear_score(desc)

    def _field_offer(self, run: dict) -> Decision:
        offer = run["field_offer"]
        new_score = self._score_wear(offer.get("description", ""))
        slots = run.get("field_slots") or []
        choices = [c["control"]["params"] for c in offer["choices"]]
        discard = next(p for p in choices if p.get("slot") == 0)
        if new_score <= 0:
            return Decision("field_pick", discard, f"skip wearable {offer['name']} (score {new_score:.1f})")
        free = [i + 1 for i, s in enumerate(slots) if not s]
        if free:
            target = next((p for p in choices if p.get("slot") == free[0]), None)
            if target:
                return Decision("field_pick", target, f"wear {offer['name']} (score {new_score:.1f})")
        worst_slot, worst_score = None, float("inf")
        for i, s in enumerate(slots):
            desc = s.get("description", "") if isinstance(s, dict) else ""
            score = self._score_wear(desc) if desc else 0.0
            if score < worst_score:
                worst_slot, worst_score = i + 1, score
        target = next((p for p in choices if p.get("slot") == worst_slot), None)
        if target and new_score > worst_score + 1:
            return Decision("field_pick", target, f"swap slot {worst_slot} ({worst_score:.1f}) for {offer['name']} ({new_score:.1f})")
        return Decision("field_pick", discard, f"keep current wearables over {offer['name']}")

    @staticmethod
    def _wear_score(desc: str) -> float:
        score = 0.0
        for name, value in re.findall(r"([一-龥]+)([+-]\d+)", desc):
            weight = next((w for key, w in WEAR_WEIGHTS.items() if name.endswith(key)), 0.3)
            score += weight * float(value)
        return score

    def greedy(self, run: dict) -> bool:
        """A doll found during the run brings the bag home on death, so extraction is never needed."""
        return self.cfg.greedy_with_doll and self._bag_item(run, "doll") is not None

    def has_healing(self, run: dict) -> bool:
        return bool(self._field_heals(run) or self._bag_heals(run, in_combat=True))

    def _advance(self, state: dict, ctr: dict) -> Decision:
        run = state["run"]
        if self.cfg.max_floor and run["floor"] >= self.cfg.max_floor and "extract" in ctr:
            return Decision("extract", {}, f"reached max floor {self.cfg.max_floor}")
        if self.cfg.extract_when_dry and "extract" in ctr and not self.has_healing(run) and not self.greedy(run):
            return Decision("extract", {}, "healing exhausted -> extract and bank the bag")
        hp_frac = run["hp"] / run["max_hp"]
        heal = self._best_heal(run, in_combat=False, force=False)
        if heal and (hp_frac < self.cfg.rush_min_hp or heal.command == "field_use"):
            return heal
        if hp_frac < self.cfg.rush_min_hp:
            forced = self._best_heal(run, in_combat=False, force=True)
            if forced:
                return forced
            if "down" in ctr:
                combat_only = any(it["id"] in COMBAT_ONLY for _, it in self._bag_heals(run, in_combat=True))
                why = "only combat-only heals left" if combat_only else "no heals"
                return Decision("down", {}, f"low HP {hp_frac:.0%}, {why}: single step so an encounter allows healing")
        if "rush" in ctr:
            risk = chain_risk(self.lasting_build(state), state, self.cal, samples=self.cfg.samples,
                              map_id=run.get("map") or "mall")
            if risk <= self.cfg.max_rush_risk:
                return Decision("rush", {}, f"rush (chain risk {risk:.1%})")
            heal = self._best_heal(run, in_combat=False, force=True)
            if heal and hp_frac < 0.95:
                heal.reason = f"rush chain risk {risk:.1%} -> {heal.reason}"
                return heal
            if "down" in ctr:
                return Decision("down", {}, f"chain risk {risk:.1%}: single step so each fight can be prepared")
            return Decision("rush", {}, f"rush (risk {risk:.1%}, no alternative)")
        return Decision("down", {}, "down")
