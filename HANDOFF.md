# Handoff — 2026-11-25 ~02:30

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
- Loadout: 2 adrenaline (+50% ATK cap) + 50 each of the 7 shop heals. Never bring doll or pouch;
  never open pouches found in a run.
- Temp gear: aprons preferred; skip the x2/x2 balance gear.
- Rush only if 4 back-to-back normal fights (buff-free stats) are <1% deadly; else single step.
- Combat-only heal (战地手术匣) top-up only below 30% HP or when the coming fight is risky.
- Healing exhausted -> extract. Fight still >10% deadly after heals/buffs -> extract.
  Exception: a doll found in the run -> greedy (no extraction), doll saves the bag on death.
- Safe box: missing collectibles first, then highest-value loot (swap if >=1.2x).
- Selling (`--sell-loot off|end|each`, default off; slow, one call per item): red/gold keep one highest-value unit per item across the warehouse;
  purple/green/white sold only from each new run's loot; blue, supplies, crates, gear kept.

## Uncommitted fixes in this checkout (after game 4: F1227, 62 bosses, died)
- `model.py`/`combat.py`: regular fights end in a draw after 15 rounds (REGULAR_ROUND_CAP); estimates no
  longer assume long fights end in a kill + kill-heal.
- `policy.py`: rush check uses permanent leech/follow-up/boss bonus only (a 虹吸滴管 +22% leech buff had
  made the F1225 rush look 0% risky; now 57%).
- `tests/test_policy.py`: 2 new tests (26 total, passing).
- Overnight farming started from this checkout with the remaining 11 free tickets (`logs/real_overnight.out`).
- Account: collection 44/48 (Tide golds done) -> only the 4 rainbows remain.

## Open items
- Local folder C:\General\Tools\Elevator not reachable from cloud sessions (Claude desktop app likely
  needs updating on Windows).
- Paid tickets (20/day, 350 gold) not enabled yet (`--buy-tickets`).
