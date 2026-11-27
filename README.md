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
of four normal fights is <1% lethal; otherwise it steps one floor at a time. With `--rush-elites on`
(default) the agent also switches the server's 自动打精英 on for rushes where fighting elites on the way is
as safe, so early floors pass without a stop at every elite (`--rush-elites off`: every rush stops at elites).

## Run & selling rules

- Healing exhausted (no potions, kits or field heals) -> extract at the next safe moment.
- A fight still >10% deadly after all heals/buffs -> extract instead of fighting.
- Rush only when 4 back-to-back normal fights (buff-free stats) are <1% deadly; otherwise single-step.
- Selling is optional and slow (one server call per item, ~10 min for a deep run's loot):
  `--sell-loot off` (default) | `end` (once after the session's last run) | `each` (after every run);
  `--sell-now` runs the red/gold pass by hand. Rules: red/gold keep one unit (highest value) per item
  across the warehouse; purple/green/white from this session's runs are sold; blue, supplies, crates,
  gear and pre-existing purple/green/white are kept.

## Components

| File | Role |
|---|---|
| `client.py` | HTTP client (login, state, commands, delta merging). |
| `model.py` | Offline game model: exact monster growth, vectorised combat (all affixes, two-way multipliers), whole-run simulation. |
| `gear.py` | Gear -> combat stats (base stats, items, 2/4-piece set bonuses, stacked affixes). |
| `optimize.py` | Searches the warehouse for the build with most bosses per run. |
| `combat.py` | Live fight estimate from the in-run state (uses `model.py`), with online-calibrated damage factors (`calibration.json`). |
| `policy.py` | `BossFarmPolicy.decide(state)` – pure decision function: fight/heal/buff/flee/extract, events, temp gear, collectible protection, rush vs. step. |
| `mdp.py` | Markov decision process of a run (floor × effective HP × decision point), solved by backward induction; the heal budget is priced with a Lagrange multiplier. |
| `markov_policy.py` | `MarkovPolicy`: `BossFarmPolicy` whose heal/fight/flee/rush/step decisions come from the solved tables (`--policy markov`). Tables are cached in `cache/`. |
| `offline.py` | Plays complete runs offline against the game's own engine (from the bot repository), for policy comparisons. |
| `trial.py` | The weekly fixed-seed trial 「来自群星的寓言集」: searches the deepest route in the local engine, then replays it live with a state check after every move. |
| `farm.py` | Runner: tickets, restocking, start/resume runs, logging (`logs/trace_*.jsonl`, `logs/runs.csv`), rainbow detection, auto-stop once all rainbows and blessing pieces are collected. |
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

Instead of `.env`, the same variables can come from the environment (e.g. a cloud environment);
`ELEVATOR_PASSWORD` is accepted as an alias of `ELEVATOR_PASS`.

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

The default loadout carries every shop heal and smoke bombs at the live carry limit (50 each) plus two
adrenaline shots. Real runs pursue two blessing sets (追寻): `--focus 6,8` picks them by number (or id,
`B06,B08`); `--focus auto` picks the two sets with most pieces still missing before every run, which
gives the most new pieces per shining drop (a missing piece weighs twice an owned one, so a set with m
of 4 missing yields a new piece with probability 2m/(m+4)). Without `--focus` the pair chosen last time
is reused, and the run is not started if there is none. Each real run logs the pursued sets and the
chance that a shining drop is new.
Fallen players' bags (遗物盒) are returned for the bounty (捞尸, 55% of their value with 专业捞尸), unless they are
worth at least `--relic-loot-value` (default 10,000,000 gold): those are looted (舔包), the safe box takes the
most valuable items, and while loot worth that much is still outside it, the agent extracts at the next safe
moment. Ordinary loot never triggers an extraction. New shining pieces are logged and, in a new `runs.csv`, written to its `shining` column (an existing file keeps its columns). `--rules legacy` plays
with the older engine's rules (elites 15 rounds without a pause, more elites on deep floors); it is only
meant for `offline.py`. Once per run the encounter odds the client shows are compared with the model (a
mismatch is logged as a warning).

## Markov policy (`--policy markov`)

The run is modelled as a Markov decision process: state = floor, effective HP (HP + shield) and the
decision point (between floors / normal or elite waiting / boss window); actions = heal with a given
item, fight, flee, single step, rush; reward = +1 per boss kill. Transitions come from the combat model
(`model.py`) and the server's encounter odds; the tables are solved backward from F4000 (~3 min, then
cached). Tables are solved for the build snapped down to a 5% grid (`MarkovConfig.bucket`), so similar
builds of different runs share one cached solve; `cache/` keeps the 20 most recently used solves. A new
solve runs in a background thread (`--solve background`, the default): the run keeps going on the
previous tables, or the threshold rules before the first ones, and switches when the solve is done.
`offline.py` uses `--solve blocking` so that comparisons stay reproducible. The heal stock is a run-long budget: instead of tracking the full inventory, each
heal unit gets a price λ, tables are solved for a grid of prices, and every decision picks the price
that matches the live stock (dual bound `min_λ V_λ + λ·H`). Consequences worth knowing:

- Extraction at a wall: once the expected further boss kills drop below `MarkovConfig.extract_below`
  (0.3), the agent extracts and banks the bag a death would lose. The estimate chains the next three bosses'
  win chances with a full heal before every window (optimistic), so it fires only when even perfect healing
  cannot get past them (with the B06 build: around F2800-2900). A boss that hopeless is not fought: the agent
  extracts (offered during a boss fight) or uses a smoke bomb, which also ends the run as an extraction.
  A doll in the bag disables this (it saves the bag anyway). Collectible banking extracts as before.
- A low-HP rush is fine: the server halts a rush before any fight at HP <= 50%, and heals are used in
  the encounter instead.
- Rush is chosen while it gives up at most `rush_tolerance` (0.002) expected bosses against single steps.
- Rushing through elites (自动打精英) is a third move in the solver: the rush fights an elite up to its pause
  while HP is above 50%, continues on a kill and stops at the pause otherwise. The agent takes it while it
  gives up at most `elite_rush_tolerance` (0.002) expected bosses against the best other move, i.e. while a
  stop before the elite (to heal, flee or smoke) would not pay. It never beats stopping; it saves requests.
  The threshold policy uses it while a four-fight chain with elites (at the floor's elite share) is <1%
  lethal. Either policy flips the setting with `rush_settings` only when it has to change.
- Elites under live rules are two segments: the first 8 rounds are valued with the best heals at the
  pause and the remaining 8 rounds after it (computed for the live elite at decision time; inside the
  solver from `kernel_stride` reference elites). The reported death risk covers the segment ahead only.

### Gold objective (`--objective gold`)

The default objective counts formal boss kills only, so the agent pushes on until a boss wall. With
`--objective gold` the same solver maximises banked gold plus `--gold-per-boss` (default 1,000,000) per
boss kill:

- A run's loot is about a thousand items worth 2k-48k gold each, so the bag's value at a floor varies
  little between runs. `mdp.LOOT_CURVES` holds it per map, measured in the game engine (mall: 0.75M at
  F100, 6.1M at F500, 12.4M at F1000, 16M at F1500). Loot counts as it is found.
- A death loses the bag; the 4-slot safe box keeps about 0.2M. Every fight's death probability is
  charged with the bag at risk: in the solver the curve's value at that floor, in live fights the
  actual bag.
- Extracting is an action between floors, in a fight and during a boss window. The agent extracts when
  no way on is worth more than banking the bag. It usually does so at a boss it cannot beat; it does not
  stop early, because a loss of the bag is avoidable until a hopeless fight.
- Tables for both objectives are cached separately (the objective is part of the cache key).

```bash
python farm.py --mode real --policy markov --objective gold --i-understand-real
```

Offline comparison against the game engine (needs a checkout of `SlimeTROH/longniang-qqbot`):

```bash
python offline.py --engine ../longniang-qqbot --runs 30 --policies threshold,markov
```

When running several offline comparisons in parallel, start each with `OPENBLAS_NUM_THREADS=1` (or 2):
the solver's matrices are small, and several processes with a BLAS thread per core starve each other
(measured: 30x slower).

## Weekly trial 「来自群星的寓言集」 (`trial.py`)

The trial (prophecy.py in the game) gives every entrant of an ISO week the same build, supplies and floors,
and seeds every roll - encounters, loot, events, fights and flee attempts - from the week, the floor and
the round. The same moves from the same state always give the same result, so the route can be searched
in the game's own engine and replayed live:

```bash
python trial.py plan --engine ../longniang-qqbot --out logs/trial
python trial.py replay --route logs/trial/route.json --i-understand-live
```

The planner keeps the farm policy for events, temp gear and field items. It decides every encounter
by beam search over flee / fight / heals and field items, scoring outcomes by HP plus the value of
what is left. It picks each star relic by playing both options through the next two bosses. When no
line survives a boss, it extracts on that floor (ranked above dying there). The replay compares the
live state with the plan after every move and stops at the first difference (`--on-divergence policy`
finishes the run with the plain trial policy instead). After a network error it resends a move only if
the server's revision did not move. The score counts the best attempt, and attempts are free.

Week 2026-W48 (黑洞视界 · 群星的低语): planned and played to F1290 (1289 cleared, extracted at a boss no
searched line survives); the next best entry stopped at F330.

## Browser userscript

`userscript/` is a Tampermonkey port of this agent that runs directly on the game page, using the
player's own logged-in session (no `.env`, no Python). Same decisions as the Python version: its tests
replay 254 recorded states and compare against Python results generated by
`userscript/scripts/gen_parity.py`. Pushing a `v*` tag publishes `elevator-farmer.user.js` as a GitHub
Release. See [userscript/README.md](userscript/README.md) (Chinese).
