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
  - Regular monsters: at most 15 rounds per fight. A draw makes the monster retreat: no loot,
    no kill-heal. A rush stops on that floor after a draw.
  - Elites (live server since 2026-11-25): at most 16 rounds, with a pause after round 8 ("第 8 回合可整备"):
    HP and the elite's state carry over and you can heal or change gear before resuming; a draw at 16 still
    makes it retreat. The enemy shows `rounds_fought`. The public bot repository still has the old rule
    (15 rounds, no pause); `--rules legacy` / `model.LEGACY_RULES` model it for offline runs.
  - Extraction gatekeepers: at most 15 rounds per fight. A draw keeps both sides' HP and round count;
    the fight resumes from there.
  - Bosses: 4 rounds per `fight`. HP and the cumulative round count carry over; between windows you can
    use potions, field items and change temp gear. A boss never retreats.
- **Rush settings** (`rush_settings`): `floors` 3/5/10, `skip_valuable`, `event_mode` (pause / random / ignore
  events) and `auto_elite`. With `auto_elite` on, a rush fights the first 8 rounds of an elite by itself and
  stops at its pause (`连战至 F628 · 精英中场`). The agent sets 10 / skip / pause / no auto-elite at run start.
- **Encounter odds** are shown for the next floor in `run.next_floor` (e.g. `📊 F247：箱34% 怪41% 精12% 奇13%｜金红68.0%(红11.9%)`).
  Since 2026-11-25 the deep-floor mix is chest 34%, normal 41%, elite 12%, event 13% (was 34 / 33 / 20 / 13:
  elites moved to normal monsters). The F1 row and the 12-floor blend are assumed unchanged.
  `farm.py` compares the shown odds with `model.floor_mix` once per run and warns on a mismatch.
- **Blessings (闻岑闪耀祝福, live since 2026-11-25).** A real run needs two pursued sets (`start` param
  `focus: [set id, set id]`, chat `追寻 1 8`; a running run that lacks them shows `heirlooms.needs_focus` and
  takes the `pursue` command). Each formal boss kill rolls once for a shining drop: 3% at F8, rising with
  depth to 10%; a drop picks one of the two sets with equal odds, missing pieces weighted higher. Pieces go
  into four independent slots (`equip_blessing {slot, piece}`, chat `祝福装配 槽位 编号`); 2 and 4 pieces of a
  set give its bonuses, single pieces only +1/+2 to one stat. State: `heirlooms.{sets, pieces, owned, slots,
  focus, needs_focus}`; set ids B01-B08. Set bonuses (2 pieces / 4 pieces):
  | Set | 2 pieces | 4 pieces |
  |---|---|---|
  | B01 错线测绘簿 | hit +14 pp (max 100%) | first missed main attack per fight re-rolls as a hit |
  | B02 末班灯列 | crit +10 pp above the cap (max 75%) | every 3rd main-attack hit on one enemy echoes 50% of its damage |
  | B03 白线压舱契 | damage reduction +10 pp (max 65%) | first hit above 10% max HP per fight: 70% of the excess removed (cap 14% max HP) |
  | B04 退件邮路 | dodge +8 pp (max 60%) | a dodge stores a ×1.45 charge for the next main-attack hit |
  | B05 余温窑账 | crit multiplier +0.35 | crits crack enemy defence −12% each (max 2) |
  | B06 缺刻值夜表 | pierce +12 pp (max 87%) | rounds 4, 8, 12, ...: act first and take ×0.65 damage (boss heavy hits too) |
  | B07 逆风放线人 | follow-up +12 pp (max 82%) | first follow-up hit per fight heals 40% of its damage (cap 12% max HP) |
  | B08 雪线留声筒 | leech +8 pp (max 50%; per-round leech heal still capped at 10% max HP) | once per run, a lethal monster hit leaves 20% max HP (before the doll) |
- Wearing blessings also brings a deep-floor armour shred ("本层祝福深层破甲", up to 60%, growing with depth):
  `player.secondary.condition.depth_shred` is the current floor's value (the formula is not known yet).
  The agent applies it to enemy defence in live fights; the solved tables do not (conservative).
  Modelled set effects: B06 2 pieces (pierce cap 87%) and 4 pieces (抢拍, rounds 4/8/... first, x0.65).
- The secondary stats (`player.secondary`) are server totals with their sources; newer servers add
  `mitigation` (打击减伤) and a `condition` block (guard, depth shred, enemy cracks, damage multiplier).
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
