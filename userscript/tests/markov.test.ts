// ============================================================================
// markov.test.ts — 价值表求解（mdp.ts）与马尔可夫策略（markov.ts）的测试（移植自 Python 版 tests/test_markov.py）
// ============================================================================

import { beforeAll, describe, expect, it } from 'vitest'
import { Calibration } from '../src/combat.ts'
import { applyPrep, liveBuild, SPECS } from '../src/field.ts'
import {
    bagAtRisk,
    bucket,
    closeBuilds,
    DEFAULT_MARKOV,
    healStock,
    isotonic,
    MarkovPolicy,
    Planner
} from '../src/markov.ts'
import type { MarkovConfig } from '../src/markov.ts'
import {
    DEFAULT_MDP,
    elitePauseKernel,
    healApply,
    healUnits,
    LOOT_CURVES,
    lootAt,
    priceIndex,
    regularKernel,
    SHOP_HEALS,
    solve,
    tableAt,
    tableRow
} from '../src/mdp.ts'
import type { MdpConfig, ValueTables } from '../src/mdp.ts'
import { enemy, fight, fightSegment, isBossFloor, makeBuild, pausesAfterSegment } from '../src/model.ts'
import type { Build, Foe } from '../src/model.ts'
import { seededRng } from '../src/rng.ts'
import { slicedSolver, yieldToEventLoop } from '../src/solver.ts'
import type { TableSolver } from '../src/solver.ts'
import type { Json } from '../src/types.ts'

// 刻意脆弱，让 F100-F200 的战斗有真实风险（实战 Build 在这么浅的楼层很安全）
const BUILD = makeBuild({
    hp: 666,
    atk: 120,
    defense: 20,
    agi: 30,
    luk: 21,
    pierce: 0.26,
    leech: 0.06,
    heavy: 0.38,
    berserk: 0.075,
    vampire: 0.1,
    critMult: 1.9
})
const SMALL: MdpConfig = {
    ...DEFAULT_MDP,
    maxFloor: 200,
    hpPoints: 26,
    bossPoints: 6,
    samples: 96,
    kernelStride: 10,
    lambdas: [0.0, 0.05, 0.3, 2.0]
}
const CAL = new Calibration({ k_player: 1.0, k_enemy: 1.0 })

let tables: ValueTables

beforeAll(() => {
    tables = solve(BUILD, 'mall', SMALL)
}, 120_000)

/** 预计还能击败的首领数可固定的策略（测试撤离规则） */
class TestPolicy extends MarkovPolicy {
    outlook: number | null = null

    protected bossOutlook(state: Json, floor: number): number {
        return this.outlook ?? super.bossOutlook(state, floor)
    }
}

function makePolicy(markov: Partial<MarkovConfig> = {}): TestPolicy {
    const policy = new TestPolicy(CAL, {}, { liveSamples: 150, mdp: SMALL, ...markov })
    policy.affixBuild = BUILD
    policy.loadoutKey = JSON.stringify([])
    policy.planner.set(tables)
    policy.rng = seededRng(7)
    return policy
}

function bagOf(...ids: string[]): Json[] {
    return ids.map((id, n) => ({ id, index: n + 1, name: id, quantity: 1 }))
}

function makeState({
    floor = 100,
    hp = 666,
    foe = null as Json | null,
    bag = [] as Json[],
    controls = ['fight', 'flee', 'extract']
} = {}): Json {
    return {
        run: {
            floor,
            hp,
            max_hp: 666,
            shield: 0,
            map: 'mall',
            effective_stats: { hp: BUILD.hp, atk: BUILD.atk, def: BUILD.defense, agi: BUILD.agi, luk: BUILD.luk },
            event: foe ? { kind: 'combat', enemy: foe } : { kind: 'ready' },
            bag: [...bag],
            field: [],
            safe: [],
            safe_capacity: 4,
            field_slots: [null, null],
            field_offer: null
        },
        player: { loadout: {}, secondary: { crit: 0.15, crit_multiplier: 1.9, pierce: 0.26, leech: 0.06 } },
        controls: controls.map(command => ({ command, params: {}, label: command })),
        atlas: { items: [] }
    }
}

function foeOf(kind: 'normal' | 'elite' | 'boss', floor: number): Json {
    const stats = enemy(kind, floor)
    return {
        ...stats,
        max_hp: stats.hp,
        name: kind,
        elite: kind !== 'normal',
        boss: kind === 'boss' ? 3 : 0,
        ambush: false,
        rounds_fought: 0
    }
}

// ------------------------------------------------------------------ 求解
describe('价值表', () => {
    it('回复药按服务器规则生效', () => {
        const at = (id: string, maxHp = 700) => [0.3, 0.8, 1.0, 1.1].map(e => healApply(SHOP_HEALS[id], e, maxHp))
        const close = (xs: number[], ys: number[]) => xs.forEach((x, i) => expect(x).toBeCloseTo(ys[i], 9))
        close(at('coffee'), [0.7, 1.0, 1.0, 1.1])
        close(at('barrier_serum'), [1.05, 1.25, 1.25, 1.25])
        close(at('critical_serum'), [1.0, 1.0, 1.0, 1.1]) // 35% 以下回复 90%
        expect(healUnits(SHOP_HEALS.triage_vial, 600)).toBeCloseTo(0.57, 9)
    })

    it('数值合理：非负、有上界、随生命单调、连战不优于单步', () => {
        const L = tables.lambdas.length
        const N = tables.grid.length
        expect(tables.ready.every(v => v >= -1e-6)).toBe(true)
        let bosses = 0
        for (let g = 2; g < SMALL.maxFloor; g++) if (isBossFloor(g)) bosses++
        expect(tableAt(tables, tables.ready, 1, 1.0)[0]).toBeLessThanOrEqual(bosses + 1e-6)
        const costly = L - 1 // 回复昂贵时生命本身有价值
        expect(tableAt(tables, tables.ready, 100, 1.0)[costly]).toBeGreaterThan(
            tableAt(tables, tables.ready, 100, 0.2)[costly]
        )
        const row = tableRow(tables, tables.ready, 100, costly)
        for (let i = 1; i < N; i++) expect(row[i] - row[i - 1]).toBeGreaterThanOrEqual(-0.02)
        // 单步提供连战的所有选项，所以连战的值不会更高
        for (let k = 1 * L * N; k < SMALL.maxFloor * L * N; k++) {
            expect(tables.rush[k]).toBeLessThanOrEqual(tables.step[k] + 1e-4)
        }
    })

    it('价格选择计入烟雾弹库存', () => {
        // 值随烟雾弹变贵而下降（直到不值得用），与求解出的表一致
        const values = Array.from(tables.smokePrices, mu => 10 - Math.min(mu, 1))
        const none = priceIndex(tables, values, 0, 0)
        const some = priceIndex(tables, values, 0, 5)
        expect(tables.smokePrices[none]).toBeGreaterThanOrEqual(1) // 没有烟雾弹：从不使用的价格
        expect(tables.smokePrices[some]).toBeLessThan(Math.max(...tables.smokePrices))
    })

    it('普通战斗转移核区分击杀与平局', () => {
        const grid = Float64Array.from({ length: 26 }, (_, i) => (1.25 * i) / 25)
        const n = grid.length
        const rowSums = (m: Float64Array) =>
            Array.from({ length: n }, (_, i) => m.subarray(i * n, (i + 1) * n).reduce((a, b) => a + b, 0))
        const rng = seededRng(5)
        let k = regularKernel(BUILD, { hp: 1e7, atk: 1, def: 0, agi: 0 }, grid, 64, rng) // 15 回合打不死
        expect(k.kill.reduce((a, b) => a + b, 0)).toBe(0)
        rowSums(k.draw)
            .slice(1)
            .forEach(s => expect(s).toBeCloseTo(1, 9))
        k = regularKernel(BUILD, { hp: 5, atk: 1, def: 0, agi: 0 }, grid, 64, rng)
        expect(k.draw.reduce((a, b) => a + b, 0)).toBe(0)
        rowSums(k.kill)
            .slice(1)
            .forEach(s => expect(s).toBeCloseTo(1, 9))
    })

    it('精英整备转移核保留每个存活者', () => {
        const grid = Float64Array.from({ length: 11 }, (_, i) => (1.25 * i) / 10)
        const enemyGrid = Float64Array.from([0.2, 0.4, 0.6, 0.8, 1.0])
        const lands = elitePauseKernel(BUILD, { hp: 1e7, atk: 1, def: 0, agi: 0 }, grid, enemyGrid, 16, seededRng(3))
        const n = grid.length
        for (let s = 0; s < n * enemyGrid.length; s++) {
            const sum = lands.subarray(s * n, (s + 1) * n).reduce((a, b) => a + b, 0)
            expect(sum).toBeCloseTo(s < enemyGrid.length ? 0 : 1, 9) // 第 0 行（已阵亡）为空
        }
    })

    it('精英战按回合上限分段', () => {
        const elite = { hp: 1, atk: 1, def: 0, agi: 0, elite: true, boss: 0, rounds_fought: 0 }
        expect(fightSegment(elite)).toEqual([1, 8])
        expect(pausesAfterSegment(elite)).toBe(true)
        expect(fightSegment({ ...elite, rounds_fought: 8 })).toEqual([9, 8])
        expect(pausesAfterSegment({ ...elite, rounds_fought: 8 })).toBe(false)
        expect(fightSegment({ ...elite, elite: false })).toEqual([1, 15])
        expect(fightSegment({ ...elite, boss: 3, rounds_fought: 5 })).toEqual([6, 3])
    })
})

// ------------------------------------------------------------------ 辅助
describe('辅助', () => {
    it('回复库存计入药品与局内回复', () => {
        const s = makeState({ bag: bagOf('coffee', 'coffee', 'field_surgery', 'adrenaline') })
        s.run.field = [{ id: 'f1', name: '速凝绷带', quantity: 2, healing: true }]
        expect(healStock(s.run)).toBeCloseTo(0.4 * 2 + 1.0 + 0.28 * 2, 9)
    })

    it('保序回归单调且保持总和', () => {
        const y = [0, 2, 1, 3, 2.5, 2.5, 5]
        const fit = isotonic(y)
        for (let i = 1; i < fit.length; i++) expect(fit[i]).toBeGreaterThanOrEqual(fit[i - 1])
        expect(fit.reduce((a, b) => a + b, 0)).toBeCloseTo(
            y.reduce((a, b) => a + b, 0),
            9
        )
        expect(Array.from(isotonic([0, 1, 2, 3, 4]))).toEqual([0, 1, 2, 3, 4])
    })

    it('校准小幅漂移时沿用同一套表', () => {
        const drifted = { ...tables.build, takenMult: tables.build.takenMult * 1.004, atk: tables.build.atk * 1.01 }
        expect(closeBuilds(tables.build, drifted, 0.1)).toBe(true)
        expect(closeBuilds(tables.build, { ...drifted, atk: tables.build.atk * 1.5 }, 0.1)).toBe(false)
    })

    it('局内道具效果按服务器规格', () => {
        const run: Json = {
            stats: { hp: 666, atk: 120, def: 20, agi: 30, luk: 21 },
            field_slots: [{ id: 'field_lamp' }, null],
            field_buffs: [],
            field_charges: [{ id: 'field_tape', left: 2 }]
        }
        const live = liveBuild(run, BUILD)
        const spot = SPECS.field_lamp.mods! // 促销聚光灯：暴击 +13%，减伤 -10%
        expect(live.critBonus).toBeCloseTo(spot.crit, 9)
        expect(live.reduction).toBeCloseTo(spot.mitigation, 9)
        expect([live.guardHits, live.guardReduction]).toEqual([2, SPECS.field_tape.reduction])
        const [boosted] = applyPrep(run, BUILD, { id: 'field_plate', name: '应急陶板', spec: SPECS.field_plate })
        expect(boosted.defense).toBeCloseTo(BUILD.defense + 0.5 * 20, 9) // 基础防御 +50%
    })

    it('防护次数减少命中伤害', () => {
        const foe: Foe = { hp: 1e9, atk: 200, def: 0, agi: 0 }
        const base = { ...BUILD, agi: 0, leech: 0, vampire: 0 }
        const lost = (b: typeof base) => {
            const res = fight(b, new Float64Array(2000).fill(1e6), new Float64Array(2000), foe, {
                boss: false,
                rounds: 1,
                rng: seededRng(3)
            })
            const hits = Array.from(res.hp).filter(h => h < 1e6)
            return hits.reduce((a, h) => a + (1e6 - h), 0) / hits.length
        }
        expect(lost({ ...base, guardHits: 1, guardReduction: 0.42 }) / lost(base)).toBeCloseTo(0.58, 1)
    })
})

// ------------------------------------------------------------------ 策略
describe('马尔可夫策略', () => {
    it('危险精英前回复，满血时直接战斗', () => {
        const policy = makePolicy()
        const low = makeState({
            hp: 120,
            foe: foeOf('elite', 100),
            bag: bagOf('coffee', 'triage_vial', 'field_surgery')
        })
        expect(policy.decide(low).command).toBe('use')
        const full = makeState({ foe: foeOf('elite', 100), bag: bagOf('coffee', 'triage_vial') })
        expect(policy.decide(full).command).toBe('fight')
    })

    it('健康时连战；低血时也连战（回复留到遭遇时）', () => {
        const policy = makePolicy()
        const controls = ['rush', 'down', 'extract']
        expect(policy.decide(makeState({ bag: bagOf('coffee'), controls })).command).toBe('rush')
        // 服务器在 HP ≤ 50% 时于战斗前停止连战，而每种回复在遭遇中都能用
        expect(policy.decide(makeState({ hp: 150, bag: bagOf('coffee', 'triage_vial'), controls })).command).toBe(
            'rush'
        )
    })

    it('回复用完也不撤离', () => {
        const s = makeState({ hp: 300, controls: ['rush', 'down', 'extract'] })
        expect(['rush', 'down']).toContain(makePolicy().decide(s).command)
    })

    it('首领窗口：低血回复，满血战斗', () => {
        const policy = makePolicy()
        const boss = foeOf('boss', 110)
        const opts = { floor: 110, foe: boss, bag: bagOf('field_surgery'), controls: ['fight', 'extract'] }
        expect(policy.decide(makeState({ ...opts, hp: 90 })).command).toBe('use')
        expect(policy.decide(makeState({ ...opts, hp: 666 })).command).toBe('fight')
    })

    it('浅层沿用阈值策略', () => {
        const s = makeState({ floor: 5, bag: bagOf('coffee'), controls: ['rush', 'down', 'extract'] })
        expect(makePolicy().decide(s).reason).toMatch(/^连战（连续风险/)
    })

    it('没有表时沿用阈值策略', () => {
        const policy = new MarkovPolicy(CAL)
        policy.affixBuild = BUILD
        policy.loadoutKey = JSON.stringify([])
        const s = makeState({ bag: bagOf('coffee'), controls: ['rush', 'down', 'extract'] })
        expect(policy.decide(s).reason).toMatch(/^连战（连续风险/)
    })

    it('冰红茶的敏捷不计入求解用的 Build', () => {
        const s = makeState({ controls: ['rush', 'down', 'extract'] })
        s.run.stats = { ...s.run.effective_stats }
        s.run.effective_stats = { ...s.run.stats, agi: BUILD.agi + 15 }
        expect(makePolicy().tableBuild(s).agi).toBe(BUILD.agi)
    })

    it('风险战斗前用防御增益，安全战斗保留增益', () => {
        // 整备后的后 8 回合、没有回复：Python 版同一状态选择应急陶板（2.00 对 1.87）
        const risky = makeState({ floor: 165, hp: 250, foe: { ...foeOf('elite', 165), rounds_fought: 8 } })
        const safe = makeState({ floor: 60, foe: foeOf('normal', 60) })
        for (const s of [risky, safe]) {
            s.run.field = [{ id: 'field_plate', name: '应急陶板', quantity: 1 }]
            s.run.stats = s.run.effective_stats
        }
        const d = makePolicy().decide(risky)
        expect([d.command, d.params]).toEqual(['field_use', { id: 'field_plate' }])
        expect(makePolicy().decide(safe).command).toBe('fight')
    })

    it('致命精英前用烟雾弹，安全战斗保留，首领前不用', () => {
        const deadly = makeState({ floor: 165, hp: 200, foe: foeOf('elite', 165), bag: bagOf('smoke', 'smoke') })
        const d = makePolicy().decide(deadly)
        expect([d.command, d.params]).toEqual(['use', { index: 1 }])
        const safe = makeState({ floor: 60, foe: foeOf('normal', 60), bag: bagOf('smoke', 'smoke') })
        expect(makePolicy().decide(safe).command).toBe('fight')
        const boss = makeState({
            floor: 190,
            hp: 150,
            foe: foeOf('boss', 190),
            bag: bagOf('smoke'),
            controls: ['fight', 'extract']
        })
        expect(makePolicy().decide(boss).command).not.toBe('use')
    })

    it('缓释再生液未结算时单步', () => {
        const s = makeState({ hp: 300, bag: bagOf('coffee'), controls: ['rush', 'down', 'extract'] })
        s.run.recovery = { left: 2, fraction: 0.16, last_floor: 99 }
        expect(makePolicy().decide(s).command).toBe('down')
    })

    it('精英整备时剩余回合危险则回复', () => {
        const elite = { ...foeOf('elite', 165), rounds_fought: 8 }
        elite.hp = Math.floor(elite.max_hp * 0.6)
        const s = makeState({
            floor: 165,
            hp: 100,
            foe: elite,
            bag: bagOf(...Array(6).fill('coffee')),
            controls: ['fight', 'flee', 'use']
        })
        expect(makePolicy().decide(s).command).toBe('use')
    })
})

// ------------------------------------------------------------------ 为了我所期待的
describe('为了我所期待的', () => {
    const wishState = (hp: number, missingInBag = false): Json => {
        const s = makeState({ floor: 170, hp, foe: foeOf('boss', 170), controls: ['fight', 'field_use'] })
        s.run.field = [{ id: 'field_wish', name: '为了我所期待的', quantity: 1 }]
        if (missingInBag) {
            s.run.bag = [{ id: 'relic_x', index: 1, name: '缺的收藏品', value: 1, can_protect: true }]
            s.atlas = { items: [{ id: 'relic_x', unlocked: false, rarity: 'gold' }] }
        }
        return s
    }

    it('首领会赢时发动，并用预览令牌确认', () => {
        const policy = makePolicy()
        const s = wishState(60)
        const d = policy.decide(s)
        expect([d.command, d.params]).toEqual(['field_use', { id: 'field_wish' }])
        s.wish_preview = { enemy: 'boss', bag: 0, safe: 0, value: 0, token: '0123456789abcdef' }
        const confirm = policy.decide(s)
        expect([confirm.command, confirm.params]).toEqual(['wish_confirm', { token: '0123456789abcdef' }])
    })

    it('安全战斗不用；不清空仍缺的收藏品', () => {
        expect(makePolicy().decide(wishState(666)).command).not.toBe('field_use')
        expect(makePolicy().decide(wishState(60, true)).params).not.toEqual({ id: 'field_wish' })
    })

    it('没有拿到预览时本层不再请求', () => {
        const policy = makePolicy()
        const s = wishState(60)
        expect(policy.decide(s).params).toEqual({ id: 'field_wish' })
        expect(policy.decide(s).params).not.toEqual({ id: 'field_wish' })
    })

    it('回满就安全的窗口不用', () => {
        const s = wishState(60)
        s.run.bag = bagOf('field_surgery')
        s.controls.push({ command: 'use', params: {}, label: 'use' })
        expect(makePolicy().decide(s).params).not.toEqual({ id: 'field_wish' })
    })
})

// ------------------------------------------------------------------ 首领墙前撤离
describe('首领墙前撤离', () => {
    it('之后的首领打不过时在层间撤离', () => {
        const policy = makePolicy()
        const s = makeState({ bag: bagOf('coffee'), controls: ['rush', 'down', 'extract'] })
        policy.outlook = 0.1
        expect(policy.decide(s).command).toBe('extract')
        policy.outlook = 2.0
        expect(policy.decide(s).command).not.toBe('extract')
    })

    it('背包里有替身娃娃时继续', () => {
        const policy = makePolicy()
        policy.outlook = 0.1
        const s = makeState({ bag: bagOf('coffee', 'doll'), controls: ['rush', 'down', 'extract'] })
        expect(policy.decide(s).command).not.toBe('extract')
    })

    it('打不过的首领：撤离，否则烟雾弹带回背包', () => {
        const wall = { ...foeOf('boss', 170), hp: 1e6, max_hp: 1e6, atk: 5000 }
        const s = makeState({
            floor: 170,
            foe: wall,
            bag: bagOf('coffee', 'smoke'),
            controls: ['fight', 'use', 'extract']
        })
        expect(makePolicy().decide(s).command).toBe('extract')
        s.controls = s.controls.filter((c: Json) => c.command !== 'extract')
        expect(makePolicy().decide(s).command).toBe('use') // 烟雾弹：同样算撤离结束本局
    })
})

// ------------------------------------------------------------------ 分片求解
describe('分片求解', () => {
    it('与一次性求解结果相同，并在求解期间让出主线程', async () => {
        const config = { ...SMALL, maxFloor: 60 }
        let ticks = 0
        const solving = { done: false }
        const ticker = (async () => {
            while (!solving.done) {
                ticks++
                await yieldToEventLoop()
            }
        })()
        const sliced = await slicedSolver(1).solve(BUILD, 'mall', config)
        solving.done = true
        await ticker
        const blocking = solve(BUILD, 'mall', config)
        expect(Array.from(sliced.ready)).toEqual(Array.from(blocking.ready))
        expect(Array.from(sliced.boss.get(50)!)).toEqual(Array.from(blocking.boss.get(50)!))
        expect(ticks).toBeGreaterThan(5)
    })
})

describe('Planner', () => {
    const fakeSolver = (fail = false) => {
        const calls: Build[] = []
        const solver: TableSolver = {
            solve: async build => {
                calls.push(build)
                if (fail) throw new Error('内存不足')
                return { ...tables, build }
            }
        }
        return { calls, solver }
    }

    it('同一 Build 只求解一次，小幅漂移沿用，明显变化时重解', async () => {
        const { calls, solver } = fakeSolver()
        const planner = new Planner(DEFAULT_MARKOV, solver)
        await Promise.all([planner.request(BUILD, 'mall'), planner.request(BUILD, 'mall')])
        await planner.request({ ...BUILD, atk: BUILD.atk * 1.02 }, 'mall')
        expect(calls).toHaveLength(1)
        expect(planner.get('mall')).not.toBeNull()
        expect(planner.get('subway')).toBeNull()
        await planner.request({ ...BUILD, atk: BUILD.atk * 1.5 }, 'mall')
        expect(calls).toHaveLength(2)
    })

    it('求解失败后不再重试，并在日志中说明', async () => {
        const { calls, solver } = fakeSolver(true)
        const logs: string[] = []
        const planner = new Planner(DEFAULT_MARKOV, solver, m => logs.push(m))
        await planner.request(BUILD, 'mall')
        await planner.request({ ...BUILD, atk: BUILD.atk * 1.5 }, 'mall')
        expect(calls).toHaveLength(1)
        expect(planner.get('mall')).toBeNull()
        expect(logs.some(m => m.includes('改用阈值规则'))).toBe(true)
    })
})

// ------------------------------------------------------------------ 连战自动打精英
describe('连战自动打精英', () => {
    it('从不优于遇精英暂停；浅层没有损失，深层停下来用烟雾弹 / 回复更好', () => {
        const LN = tables.lambdas.length * tables.grid.length
        let shallow = 0
        let deep = 0
        for (let k = LN; k < SMALL.maxFloor * LN; k++) {
            const loss = tables.rush[k] - tables.rushElite[k]
            expect(loss).toBeGreaterThanOrEqual(-1e-4)
            if (k < 60 * LN) shallow = Math.max(shallow, loss)
            else if (k >= 100 * LN) deep = Math.max(deep, loss)
        }
        expect(shallow).toBeLessThan(1e-3)
        expect(deep).toBeGreaterThan(0.05)
    })

    const rushReady = (flag: boolean): Json => {
        const s = makeState({ floor: 40, bag: bagOf('coffee'), controls: ['rush', 'down', 'extract'] })
        s.player.rush_auto_elite = flag
        return s
    }

    for (const [loss, auto] of [
        [0.001, true],
        [0.01, false]
    ] as const) {
        it(`损失 ${loss} 时${auto ? '开启' : '关闭'}「自动打精英」`, () => {
            const LN = tables.lambdas.length * tables.grid.length
            const worse = { ...tables, rushElite: tables.rushElite.slice() }
            for (let k = 40 * LN; k < 41 * LN; k++) worse.rushElite[k] -= loss
            const policy = makePolicy()
            policy.planner.set(worse)
            const d = policy.decide(rushReady(!auto))
            expect([d.command, d.params]).toEqual(['rush_settings', { auto_elite: auto }])
            expect(policy.decide(rushReady(auto)).command).toBe('rush')
        })
    }

    it('关闭选项后求解与决策都不含此项', () => {
        const policy = new MarkovPolicy(CAL, { rushElites: false })
        expect(policy.mcfg.mdp.rushElites).toBe(false)
    })
})

describe('价值表网格', () => {
    it('保守吸附、相近 Build 共用、幂等', () => {
        const a = bucket(BUILD, 0.05)
        expect(a.atk).toBeLessThanOrEqual(BUILD.atk)
        expect(a.hp).toBeLessThanOrEqual(BUILD.hp)
        expect(a.takenMult).toBeGreaterThanOrEqual(BUILD.takenMult)
        expect(a.atk).toBeGreaterThan(BUILD.atk / 1.05)
        expect(Number.isInteger(a.hp) && Number.isInteger(a.atk)).toBe(true)
        const near = { ...a, atk: a.atk * 1.03, hp: a.hp + 3, takenMult: a.takenMult * 0.98 }
        expect(bucket(near, 0.05)).toEqual(a)
        expect(bucket(a, 0.05)).toEqual(a)
        expect(bucket(BUILD, 0)).toEqual(BUILD)
    })
})

// ------------------------------------------------------------------ 金币目标
describe('金币目标', () => {
    const WEAK = { ...BUILD, hp: 400, defense: 5, atk: 90, leech: 0, vampire: 0 } // 约 F70 起打不过
    let gold: ValueTables
    beforeAll(() => {
        gold = solve(WEAK, 'mall', { ...SMALL, gold: true, bossValue: 0.2 })
    }, 120_000)

    it('物资曲线：插值，最后一点之后沿最后一段斜率延伸；未实测地图用商场曲线', () => {
        expect(lootAt('mall', 0)).toBe(0)
        expect(lootAt('mall', 100)).toBeCloseTo(0.75, 9)
        expect(lootAt('mall', 150)).toBeCloseTo((0.75 + 1.71) / 2, 9)
        const pts = LOOT_CURVES.mall
        const [[x1, y1], [x2, y2]] = pts.slice(-2)
        expect(lootAt('mall', x2 + 100)).toBeCloseTo(y2 + (100 * (y2 - y1)) / (x2 - x1), 9)
        expect(lootAt('unknown', 300)).toBe(lootAt('mall', 300))
        expect(lootAt('zenith', 100)).toBeGreaterThan(lootAt('mall', 100))
    })

    it('从不低于撤离，越深的首领越常撤离', () => {
        const L = gold.lambdas.length
        const N = gold.grid.length
        for (let g = 1; g < SMALL.maxFloor; g++) {
            for (let l = 0; l < L; l++) {
                for (let i = 1; i < N; i++) expect(gold.ready[(g * L + l) * N + i]).toBeGreaterThanOrEqual(-1e-5)
            }
        }
        const NB = gold.bossGrid.length
        const costly = L - 1
        const zeros = [30, 90, 150, 190].map(f => {
            const tbl = gold.boss.get(f)!.subarray(costly * N * NB, (costly + 1) * N * NB) // 第 1 窗口
            let n = 0
            for (let k = NB; k < tbl.length; k++) if (tbl[k] <= 1e-6) n++
            return n
        })
        expect(zeros[0]).toBe(0)
        expect([...zeros].sort((a, b) => a - b)).toEqual(zeros)
        expect(zeros[3]).toBeGreaterThan(50)
    })

    it('物资计入价值，首领目标不变', () => {
        const withGold = solve(BUILD, 'mall', { ...SMALL, gold: true })
        const L = tables.lambdas.length
        const N = tables.grid.length
        const full = N - 6 // 生命 1.0（26 个格点到 1.25）
        expect(withGold.ready[L * N + full] - tables.ready[L * N + full]).toBeCloseTo(lootAt('mall', 201), 1)
    })

    it('撤离而不是硬打打不过的首领', () => {
        const policy = makePolicy({ extractBelow: 0, mdp: { ...SMALL, gold: true, bossValue: 0.2 } })
        policy.affixBuild = WEAK
        policy.planner.set(gold)
        const s = makeState({
            floor: 190,
            hp: 400,
            foe: foeOf('boss', 190),
            bag: [{ id: 'r05', index: 9, name: '红', value: 48_000, quantity: 30 }],
            controls: ['fight', 'extract']
        })
        s.run.max_hp = 400
        s.run.effective_stats = { hp: 400, atk: 90, def: 5, agi: WEAK.agi, luk: WEAK.luk }
        const d = policy.decide(s)
        expect(d.command).toBe('extract')
    })

    it('背包才计入死亡损失', () => {
        const run = { bag: [{ value: 48_000, quantity: 10 }, { value: 2_000 }], safe: [{ value: 48_000 }] }
        expect(bagAtRisk(run)).toBeCloseTo(0.482, 9)
    })
})
