# Handoff — 2026-11-25 ~13:15

## State
- Repo `Xc165543337/elevator_script` (main @ e28ff28) is the master copy. It already contains all agent
  changes from the session; the local working copy was behind and has been synced from it.
- Repo-only fixes vs the old local copy:
  - `policy.py`: rush safety check gets the run's map (`map_id=run.get("map") or "mall"`); without it the
    check used Mall monsters on every map (dangerous on Tide/Zenith).
  - `requirements.txt`: adds `numpy` (needed by model.py/combat.py).
  - `.gitignore`: `.claude/`. `README.md`: userscript section.
- Account after games 1-3: 0/4 rainbows, collection 41/48, gold ~61M, 14 free tickets left that day.
- Game 4 (Mall) was still running at shutdown: F1051, 54 bosses. If the process was stopped, the run stays
  on the server and expires after 30 min of inactivity; `python farm.py --mode real --runs 1
  --i-understand-real` resumes it.

## Results (real Mall runs)
| Run | Floor | Bosses | End |
|---|---|---|---|
| first | 1159 | 59 | died (rush without healing; fixed) |
| game 1 | 740 | 38 | extracted early by mistake (sim/real guard added) |
| game 2 | 1193 | 61 | died after F1190 boss; doll kept the bag |
| game 3 | 1164 | 59 | died: heals used up + expiring buffs in rush check (both fixed) |
| game 4 | 1051+ | 54+ | running |

## Current rules (all in code)
- Gear: keep current loadout (#1839, #11655, #5522, #12248). Affixes do not stack (verified); no rerolls.
- Loadout: 2 adrenaline (+50% ATK cap) + 50 (the live per-supply carry limit) each of the 7 shop heals and smoke bombs. Never bring doll or pouch;
  never open pouches found in a run.
- Temp gear: aprons preferred; skip the x2/x2 balance gear.
- Rush only if 4 back-to-back normal fights (buff-free stats) are <1% deadly; else single step.
- Combat-only heal (战地手术匣) top-up only below 30% HP or when the coming fight is risky.
- Healing exhausted -> extract. Fight still >10% deadly after heals/buffs -> extract.
  Exception: a doll found in the run -> greedy (no extraction), doll saves the bag on death.
- Safe box: missing collectibles first, then highest-value loot (swap if >=1.2x).
- Selling (`--sell-loot off|end|each`, default off; slow, one call per item): red/gold keep one highest-value unit per item across the warehouse;
  purple/green/white sold only from each new run's loot; blue, supplies, crates, gear kept.

## Round-cap / leech fixes (committed on this branch)
- `model.py`/`combat.py`: regular fights end in a draw after 15 rounds (REGULAR_ROUND_CAP).
- `policy.py`: rush check uses permanent leech/follow-up/boss bonus only.
- After game 4 (F1227, 62 bosses, died): collection 44/48 -> only the 4 rainbows remain.

## Markov decision process policy (2026-11-25, branch `fix/round-cap-and-lasting-leech`)
- `mdp.py`: run MDP (floor x effective HP x decision point), backward induction F4000 -> F1, heal budget
  priced with a Lagrange multiplier (18 prices solved at once). ~65 s per build, cached in `cache/`.
- `markov_policy.py`: `MarkovPolicy` = BossFarmPolicy + MDP decisions for heal / fight / flee / rush / step,
  live fight simulation against the actual enemy, field buffs / grenades / buff pairs valued in bosses.
- `field.py` + `data/field_items.json`: server field-item specs; effects the state only exposes as ids.
- `offline.py`: full runs against the game author's engine (SlimeTROH/longniang-qqbot) in-process.
  Validated: one offline threshold run F1167 / 59 bosses (real runs 59-62); learned damage factors
  1.000 / 0.981 vs live 1.004 / 0.967.
- `farm.py --policy markov` selects it; default stays `threshold`.
- Model fixes from the data: floor encounter mix follows the server curve (was flat 23% / 14%); a rush
  halts after a regular-fight draw.
- Offline results (old engine, bosses per run): threshold 68, 79, 61, 73, 71, 57, 61 (mean 67.1) and one
  run with 50 smokes: 54. Markov before the rush/regen fix 62.9 +- 1.9 (10 runs); after it 79, 78, 83
  (mean 80, deaths only once the shop heals ran out around F1550); one run with 50 smokes: 47 (a 0.4%
  elite fight lost at F918). Single runs are noisy.

## Server update of 2026-11-25 (adapted on this branch)
- Elites: 16 rounds with a pause after round 8 (heal / change gear). `model.ServerRules`, `fight_segment`;
  the MDP values the pause (`mdp.elite_pause_kernel`), the live policy evaluates it per elite, the
  threshold estimate covers the segment ahead. `--rules legacy` keeps the old rule for `offline.py`.
- Blessings (闻岑闪耀祝福): real runs need two pursued sets (`--focus`, default = last choice, currently
  B06 + B08). Shining drops are logged. Set bonuses are listed in GAME_NOTES.md; none are worn yet
  (one piece each of B06 and B08 owned), so they are not modelled. Once 2+ pieces of a set are worn, the
  model needs: B04 dodge and B01 hit (the model computes dodge / hit from agility itself), the 4-piece
  effects (B06 抢拍 is large against boss heavy hits).
- Encounter odds: deep floors are now normal 41% / elite 12% / event 13% / chest 34% (was 33 / 20 / 13 / 34),
  in `ServerRules.deep_mix`. `farm.py` checks `run.next_floor` once per run and
  warns if the server's odds differ from the model again.
- `player.secondary.mitigation` (打击减伤) is read into the build (a server total, field items included).
- Not verified on a live elite yet: the field names of the pause state (`rounds_fought` = 8, controls).
  Watch the first live elite in `--verbose` output.
- The userscript port follows (pursuit, elite segment, mitigation, shining log). Its parity fixture was
  not regenerated: the Python fight engine has moved on (field-item build fields, follow-up order) and the
  TS port has not, which a regenerated fixture exposes (2 failing parity tests). Separate follow-up.

## First live run with the Markov policy (2026-11-25 13:26-13:53)
- Resumed the user's real Mall run at F616 (restarted once at F685 to pick up fixes). 29 bosses
  (F630-F1190), died at F1207 in a rush: a F1206 normal left 382/700 HP (above the 50% rush stop), the
  F1207 normal won with 34 HP left. Model: 3.6% death for that two-fight rush, 16.8% from 382 HP;
  rush 1.15 vs step 1.16 was inside rush_tolerance 0.02, now 0.002. ~45 heal units were left unused.
- Shining drops: 缺齿校时轮 x2, 早一秒交班条 -> B06 缺刻值夜表 4/4 collected (none worn yet).
- Confirmed live: elite pause state (`rounds_fought` 8, enemy HP kept), rush stop `精英中场`, encounter
  odds 41/12/13/34, rush settings can be changed during a run.

## Real runs with the full B06 set (2026-11-25 14:00-15:52)
- Run 1 (Markov, full loadout, pursuit B08 + B03): 111 bosses by F2150 under the agent, F2110 at 106.
  The model expected ~39 more bosses at F1010 (vs ~4.6 before the set): 抢拍 and the deep-floor shred
  (fitted: 0.6 x d / (d + 85), d = floor - 50) make deep floors far easier. Shining: B08 4/4, B03 1/4.
- At the F2150 boss the agent picked the wish (为了我所期待的), which would have wiped 23.5M of run loot.
  The margin came from heal costs (~14 windows of full heals), not risk: from full HP a window cannot
  kill (worst ~638 of 702). The user healed through manually and reached F2176. The wish now requires
  the segment to stay >= 5% deadly after the best heal at hand, and its loot limit follows the 10M relic
  threshold.
- Live API details learned: the wish preview (token) comes in the command result, not the state; the
  server can return HTTP 502 while redeploying (the agent now waits it out).

## Live run with the Markov policy (how to start)
- Credentials: the cloud environment now has `ELEVATOR_USER` and `ELEVATOR_PASSWORD` (accepted as alias of
  `ELEVATOR_PASS`). They are loaded only in sessions started after they were added.
- One active run per account: make sure no other agent (e.g. the local Windows machine) is playing,
  otherwise `farm.py` resumes that run and two agents drive it.
- Order: `python farm.py --mode sim --policy markov --runs 1 --verbose` (free; first contact of the new code
  with the live server, whose state may differ from the old engine), then
  `python farm.py --mode real --policy markov --runs 1 --i-understand-real` (add `--focus 6,8` to change the
  pursued blessing sets).
- Expect 2-3 pauses of ~1 min in the first run (table solves), cached afterwards.

## Open items
- Local folder C:\General\Tools\Elevator not reachable from cloud sessions (Claude desktop app likely
  needs updating on Windows).
- Paid tickets (20/day, 350 gold) not enabled yet (`--buy-tickets`).
