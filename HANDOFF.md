# Handoff — 2026-11-25 ~10:20

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
- `mdp.py`: run MDP (floor x effective HP x decision point), backward induction F2600 -> F1, heal budget
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
- Offline comparison (threshold 20 runs, then markov 10 runs) was running when this was written; first
  threshold runs: 68, 79, 61, 73 bosses. Results not in yet.

## Live run with the Markov policy (not started yet)
- Credentials: the cloud environment now has `ELEVATOR_USER` and `ELEVATOR_PASSWORD` (accepted as alias of
  `ELEVATOR_PASS`). They are loaded only in sessions started after they were added.
- One active run per account: make sure no other agent (e.g. the local Windows machine) is playing,
  otherwise `farm.py` resumes that run and two agents drive it.
- Order: `python farm.py --mode sim --policy markov --runs 1 --verbose` (free; first contact of the new code
  with the live server, whose state may differ from the old engine), then
  `python farm.py --mode real --policy markov --runs 1 --i-understand-real`.
- Expect 2-3 pauses of ~1 min in the first run (table solves), cached afterwards.

## Open items
- Local folder C:\General\Tools\Elevator not reachable from cloud sessions (Claude desktop app likely
  needs updating on Windows).
- Paid tickets (20/day, 350 gold) not enabled yet (`--buy-tickets`).
