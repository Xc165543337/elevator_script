# Elevator Boss Farmer

An automated agent for **无尽电梯 (Endless Elevator)** that farms bosses to complete the item collection.
Rainbow (虹彩) items drop at a flat **0.5% per formal boss kill** on every map, so the agent maximizes
boss kills per ticket and per hour on the easiest map (废弃商场 / `mall`).

## Strategy

| Fact | Consequence |
|---|---|
| Rainbow drop rate is the same on every map; drops go straight to the warehouse (kept even on death). | Farm the easiest map (`mall`) and go as deep as possible. |
| Bosses at F8, F18, F30, then every 20 floors; killing one fully heals you. | One ticket can yield 30–50+ bosses (F570 naive → F1030+ with this policy, in practice mode). |
| Runs only expire after 30 min of **inactivity**. | No limit on run length. |
| A regular fight resolves fully in one call; bosses fight in 4-round windows. | Decide fight / heal / flee *before* engaging, using a Monte Carlo fight simulator. |
| Rush (连战) auto-descends up to 10 floors and stops at elites, bosses, events. | Few API calls per floor; every risky fight gets a decision point. |

Expected effort: all 4 rainbows need ≈ 8.3 drops ≈ **1,670 boss kills** on average (coupon collector).
Measured cost: ~8 s/boss at F8–200, ~17 s at F400–600, ~27 s beyond F800.

The three Tide Power Station collectibles (归潮档案) only drop on `tide`: run a few Tide runs; the agent
locks any still-missing collectible into the safe box immediately (kept on death) or extracts to bank it.

## Build & temp-gear findings

Affixes do **not** stack across items (verified in game: 3x 破甲 still shows 26% pierce), so the
current loadout (#1839, #11655, #5522, #12248) is already the best combination in the warehouse
(optimizer, 400 simulated runs per build). Rerolling the dead affixes into 反甲刺猬/孤注一掷 adds
at most ~3 bosses/run; carrying 50 instead of 30 surgery kits adds ~2.

| Setup (current gear) | Bosses / run |
|---|---|
| apron x2 temp gear + 2 adrenaline on F1 (model) | 46-48 |
| balance x2 (x4 dealt / x4 taken) (model) | 26 |
| **First real run** (50 surgery kits) | **59 (F1159, 13.6 min)** |

Limits: event ATK bonus <= +60%, adrenaline <= +50% (2 shots, separate pool), 3 floor-limited
buffs, 2 temp-gear slots, 2 grenades per boss. Two balance gears = x4 damage taken (measured).
Rush auto-fights normal monsters without healing, so deep down the agent only rushes when a chain
of three normal fights is <1% lethal; otherwise it steps one floor at a time.

## Run & selling rules

- Healing exhausted (no potions, kits or field heals) -> extract at the next safe moment.
- A fight still >10% deadly after all heals/buffs -> extract instead of fighting.
- Rush only when 4 back-to-back normal fights (buff-free stats) are <1% deadly; otherwise single-step.
- With `--sell-loot`: red/gold keep one unit (highest value) per item across the warehouse; purple/green/white
  from each new run are sold; blue, supplies, crates, gear and pre-existing purple/green/white are kept.

## Components

| File | Role |
|---|---|
| `client.py` | HTTP client (login, state, commands, delta merging). |
| `model.py` | Offline game model: exact monster growth, vectorised combat (all affixes, two-way multipliers), whole-run simulation. |
| `gear.py` | Gear -> combat stats (base stats, items, 2/4-piece set bonuses, stacked affixes). |
| `optimize.py` | Searches the warehouse for the build with most bosses per run. |
| `combat.py` | Live fight estimate from the in-run state (uses `model.py`), with online-calibrated damage factors (`calibration.json`). |
| `policy.py` | `BossFarmPolicy.decide(state)` – pure decision function: fight/heal/buff/flee/extract, events, temp gear, collectible protection, rush vs. step. |
| `farm.py` | Runner: tickets, restocking, start/resume runs, logging (`logs/trace_*.jsonl`, `logs/runs.csv`), rainbow detection, auto-stop when all rainbows are unlocked. |
| `tests/` | Unit tests for the pure logic (`python -m pytest -q tests`). |
| `docs/` | Game formulas, in-game guide dump, notes (`GAME_NOTES.md`). |

## Setup (Windows / Linux)

```bash
python -m pip install -r requirements.txt
```

Create `.env` next to `farm.py` (never commit it):

```
ELEVATOR_USER=your_account_or_qq
ELEVATOR_PASS=your_password
```

## Usage

```bash
python farm.py --status                                   # collection progress
python farm.py --mode sim --runs 1                        # free practice run, no rewards
python farm.py --mode real --runs 20 --i-understand-real  # farm with free tickets
python farm.py --mode real --runs 40 --buy-tickets --i-understand-real   # also buy the 20 paid tickets
python farm.py --mode real --map tide --runs 3 --max-floor 40 --i-understand-real  # Tide collectibles
```

Useful options: `--loadout field_surgery=12,critical_serum=8,triage_vial=10,coffee=10`,
`--gold-reserve 1000000`, `--max-floor N`, `--delay 0.05`, `--balance-gear-score N`, `--no-restock`, `--verbose`.
An interrupted run is resumed automatically on the next start.

## Browser userscript

`userscript/` is a Tampermonkey port of this agent that runs directly on the game page, using the
player's own logged-in session (no `.env`, no Python). Same decisions as the Python version: its tests
replay 254 recorded states and compare against Python results generated by
`userscript/scripts/gen_parity.py`. Pushing a `v*` tag publishes `elevator-farmer.user.js` as a GitHub
Release. See [userscript/README.md](userscript/README.md) (Chinese).
