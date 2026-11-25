# Endless Elevator (无尽电梯) — Game Notes

Extraction-style roguelite ("search, fight, extract"). Shared save with a QQ-group bot.

## API (reverse-engineered from the web client)

Base: `https://elevator.slime.icu/elevator/api` — requests must carry `Origin`/`Referer` of the site.

| Endpoint | Method | Purpose |
|---|---|---|
| `/auth/login` | POST `{username,password,remember}` | Cookie session |
| `/game/state` | GET | Full state (~440 KB) + `sync_token` |
| `/game/command` | POST `{command, params, revision}` | Structured game moves; returns `result` + `delta`/`state` |
| `/game/action` | POST `{action, argument, revision}` | Text-style queries (help, lookup, quests…) |

The state always lists the legal moves: `state.controls` (structured) and `state.actions`.
A pending temp-gear offer (`run.field_offer`) blocks other moves until `field_pick` is sent.

### Commands seen
- Lobby: `start {map_id, supplies:[{id,quantity}]}`, `simulate {map_id}`, `buy`, `sell`, `craft`, `rush_settings`
- In run: `down`, `rush` (auto-descend N floors), `fight`, `flee`, `extract`, `event {option}`, `leave`,
  `use {index}`, `field_use`, `field_pick {id,floor,slot}`, `save_box {index}`, `swap_box`, `flare`, `rob`, `rescue`

## Core loop
1. Pick a map (+ entry fee) and supplies; costs 1 ticket (20 free/day, up to 20 paid).
2. Each floor rolls: chest / monster / event (odds shown in the option text) / relic of a fallen player.
3. `down` to go deeper (higher loot value and gold/red drop chance), or `extract` to bank the bag.
4. Death loses the temporary bag; items locked in the **safe box** (4 slots) are kept.
5. Bosses at F8, F18, F30, then every 20 floors. Fights resolve in 4-round windows (heavy hit ×1.26 every 4th round); healing allowed between windows. Killing a boss fully heals.

## Mechanics that matter for an agent
- **A regular `fight` resolves to the end** (victory or death) in one call → decide fight / flee / heal *before* engaging.
- **Round caps** (confirmed in the server code):
  - Regular monsters and elites: at most 15 rounds per fight. A draw makes the monster retreat: no loot,
    no kill-heal. A rush stops on that floor after a draw.
  - Extraction gatekeepers: at most 15 rounds per fight. A draw keeps both sides' HP and round count;
    the fight resumes from there.
  - Bosses: 4 rounds per `fight`. HP and the cumulative round count carry over; between windows you can
    use potions, field items and change temp gear. A boss never retreats.
- Flee chance is shown in the control label (e.g. `脱战 · 64%`) and drops with depth. Bosses and extraction ambushers cannot be fled normally.
- Extraction ambush on harder maps (e.g. Zenith 39% from F4) unless the floor's monster was killed.
- Full combat/drop/monster-growth formulas: `docs/formulas_zh.txt`. Item/map/affix reference: `docs/game_guide_zh.txt`.
- **Simulation mode** (`simulate`): free, uses real gear/stats, grants a starter kit, no loot kept, not ranked. Safe sandbox for testing policies.

## Current character (at time of analysis)
HP 666 · ATK 149 · DEF 69 · AGI 77 · LUK 21; all-red gear; pierce 26%, dodge 33.8%, crit 15.5% (×1.9), leech 6%.
Gold ≈ 34.7M, 18 tickets, safe box 4 slots, all 6 maps unlocked.

## Observed difficulty (simulation, naive policy)
- Mall (×1.0): one-shots everything through F12, including the F8 boss.
- Zenith (×4.45): cleared to F57 (F50 boss beaten with 2 heals); died at F58 to an elite (1526 HP) entering at 61% HP.
  Bag value at death ≈ 1.31M (simulated).
