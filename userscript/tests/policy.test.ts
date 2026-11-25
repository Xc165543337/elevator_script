// ============================================================================
// policy.test.ts — 决策与引擎的单元测试（移植自 Python 版 tests/test_policy.py）
// ============================================================================

import { describe, expect, it } from 'vitest'
import { Calibration, parseBattleLog } from '../src/combat.ts'
import { enemy, fight, makeBuild, withTempGear } from '../src/model.ts'
import { BossFarmPolicy } from '../src/policy.ts'
import type { PolicyConfig } from '../src/policy.ts'
import { seededRng } from '../src/rng.ts'
import type { Json } from '../src/types.ts'

const CAL = new Calibration({ k_player: 1.0, k_enemy: 1.0 })
const AFFIXES = makeBuild({
    hp: 666,
    atk: 150,
    defense: 70,
    agi: 77,
    luk: 21,
    pierce: 0.26,
    leech: 0.06,
    heavy: 0.38,
    berserk: 0.075,
    vampire: 0.1,
    critMult: 1.9
})

function makePolicy(cfg: Partial<PolicyConfig> = {}): BossFarmPolicy {
    const policy = new BossFarmPolicy(CAL, cfg)
    policy.affixBuild = AFFIXES
    policy.loadoutKey = JSON.stringify([])
    return policy
}

interface StateOpts {
    hp?: number
    enemy?: Json
    bag?: Json[]
    field?: Json[]
    safe?: Json[]
    missing?: string[]
    controls?: string[]
}

function makeState({
    hp = 600,
    enemy: foe,
    bag = [],
    field = [],
    safe = [],
    missing = [],
    controls = ['fight', 'extract']
}: StateOpts = {}): Json {
    return {
        run: {
            floor: 100,
            hp,
            max_hp: 666,
            shield: 0,
            effective_stats: { hp: 666, atk: 150, def: 70, agi: 77, luk: 21 },
            event: foe ? { kind: 'combat', enemy: foe } : { kind: 'ready' },
            bag,
            field,
            safe,
            safe_capacity: 4,
            field_slots: [null, null],
            field_offer: null
        },
        player: { loadout: {}, secondary: { crit: 0.15, crit_multiplier: 1.9, pierce: 0.26, leech: 0.06 } },
        controls: controls.map(c => ({ command: c, params: {}, label: c })),
        atlas: { items: missing.map(m => ({ id: m, unlocked: false, rarity: 'gold' })) }
    }
}

const deathRate = (hp: Float64Array) => hp.filter(v => v <= 0).length / hp.length

it('解析战斗日志：排除暴击，计入护盾吸收', () => {
    const log = [
        'R1：你打 179（怪剩0）',
        'R2：你暴击打 296（怪剩361） ｜ 吸血 +17 ｜ 怪打 49（你剩489，盾挡3）',
        'R3：你出手被怪闪开 ｜ 怪出手被你闪开'
    ]
    const hits = parseBattleLog(log)
    expect(hits.player).toEqual([[1, 179]])
    expect(hits.enemy).toEqual([[2, 52]])
})

it('战斗引擎：弱敌安全、强敌致命', () => {
    const rng = seededRng(1)
    const hp = new Float64Array(500).fill(600)
    const shield = new Float64Array(500)
    const weak = { hp: 50, atk: 10, def: 2, agi: 5 }
    const strong = { hp: 5000, atk: 600, def: 40, agi: 30 }
    expect(deathRate(fight(AFFIXES, hp, shield, weak, { boss: false, rng }).hp)).toBe(0)
    expect(deathRate(fight(AFFIXES, hp, shield, strong, { boss: false, rng }).hp)).toBeGreaterThan(0.9)
})

it('怪物成长与实战首领一致', () => {
    expect(enemy('boss', 1030).hp).toBe(4067) // 实战练习局中观测
    expect(enemy('normal', 2)).toEqual({ hp: 45, atk: 11, def: 2, agi: 7 })
})

it('两件平衡性调整临装：承受伤害 ×4', () => {
    const rng = seededRng(2)
    const foe = { hp: 1e9, atk: 100, def: 0, agi: 0 }
    const base = { ...AFFIXES, agi: 0, leech: 0, vampire: 0 }
    const start = new Float64Array(400).fill(1e6)
    const taken = (hp: Float64Array) => {
        const hits = [...hp].filter(v => v < 1e6)
        return hits.reduce((s, v) => s + (1e6 - v), 0) / hits.length
    }
    const one = fight(base, start, new Float64Array(400), foe, { boss: false, rounds: 1, rng }).hp
    const four = fight(withTempGear(base, 'balance', 'balance'), start, new Float64Array(400), foe, {
        boss: false,
        rounds: 1,
        rng
    }).hp
    const ratio = taken(four) / taken(one)
    expect(ratio).toBeGreaterThan(3.8)
    expect(ratio).toBeLessThan(4.2)
})

it('首层使用两支肾上腺素后开始连战', () => {
    const state = makeState({
        hp: 666,
        bag: [{ id: 'adrenaline', index: 1, name: '暴走肾上腺素' }],
        controls: ['rush']
    })
    state.run.floor = 1
    const policy = makePolicy()
    expect(policy.decide(state).command).toBe('use')
    expect(policy.decide(state).command).toBe('use') // 第二层
    expect(policy.decide(state).command).toBe('rush') // 已达上限 2
})

it('弱敌直接战斗', () => {
    const state = makeState({ enemy: { name: 'mob', hp: 50, atk: 10, def: 2, agi: 5 } })
    expect(makePolicy().decide(state).command).toBe('fight')
})

it('危险战斗前先回血', () => {
    const foe = { name: 'elite', elite: true, hp: 1500, atk: 260, def: 45, agi: 24 }
    const bag = [{ id: 'field_surgery', index: 3, name: '战地手术匣' }]
    const d = makePolicy().decide(makeState({ hp: 200, enemy: foe, bag }))
    expect(d.command).toBe('use')
    expect(d.params).toEqual({ index: 3 })
})

it('无望且资源耗尽时撤离', () => {
    const foe = { name: 'elite', elite: true, hp: 5000, atk: 600, def: 40, agi: 30 }
    expect(makePolicy().decide(makeState({ hp: 200, enemy: foe })).command).toBe('extract')
})

it('把仍缺的图鉴物品锁入安全箱', () => {
    const bag = [{ id: 'tide_zero_rule', index: 7, name: '零潮水位原尺', can_protect: true }]
    const d = makePolicy().decide(makeState({ bag, missing: ['tide_zero_rule'], controls: ['rush'] }))
    expect(d.command).toBe('save_box')
    expect(d.params).toEqual({ index: 7 })
})

it('奇遇评分偏好稳妥收益', () => {
    const score = BossFarmPolicy.scoreOption
    expect(score('敏捷检定94%成功→得1件紫色底起跳物资；失败(6%)扣约133HP', 1.0, 666)).toBeLessThan(0)
    expect(score('100%→得十倍浓缩冰红茶', 1.0, 666)).toBeGreaterThan(0)
    expect(score('耗 20% 当前血(约101)，100%→本局攻+20%', 0.9, 666)).toBeGreaterThan(0)
})

it('临装评分', () => {
    expect(BossFarmPolicy.wearScore('本局佩戴，占 2 个临装位之一 · 闪避率+11% · 攻击+13%')).toBeGreaterThan(20)
    expect(BossFarmPolicy.wearScore('本局佩戴 · 减伤+17% · 攻击-8%')).toBeGreaterThan(0)
})

it('被拒绝的操作不再重复', () => {
    const foe = { name: 'elite', elite: true, hp: 1500, atk: 260, def: 45, agi: 24 }
    const field = [{ id: 'field_caster', name: 'buff', description: '持续 3 层 · 攻击+24%', quantity: 1 }]
    const policy = makePolicy()
    const state = makeState({ hp: 150, enemy: foe, field })
    const first = policy.decide(state)
    expect(first.command).toBe('field_use')
    policy.ban(state.run.floor, first)
    expect(policy.decide(state).command).not.toBe('field_use')
})

const KIT = [{ id: 'field_surgery', index: 4, name: '战地手术匣' }]

it('深层低血量时单步前进而不连战', () => {
    const state = makeState({ hp: 300, bag: KIT, controls: ['rush', 'down', 'extract'] })
    state.run.floor = 1155
    const d = makePolicy().decide(state)
    expect(d.command, d.reason).toBe('down')
})

it('浅层满血时连战', () => {
    const state = makeState({ hp: 666, bag: KIT, controls: ['rush', 'down', 'extract'] })
    state.run.floor = 40
    expect(makePolicy().decide(state).command).toBe('rush')
})

it('锁入最值钱的战利品，补给留在背包', () => {
    const bag = [
        { id: 'coffee', index: 1, name: '冰美式', value: 300, can_protect: true },
        { id: 'r08', index: 2, name: 'red', value: 35000, can_protect: true },
        { id: 'g16', index: 3, name: 'gold', value: 4000, can_protect: true }
    ]
    const d = makePolicy().decide(makeState({ bag, controls: ['rush'] }))
    expect(d.command).toBe('save_box')
    expect(d.params).toEqual({ index: 2 })
})

it('只在明显更值钱时替换安全箱物品', () => {
    const safe = [1, 2, 3, 4].map(i => ({ id: `x${i}`, index: i, name: 'x', value: 10000 }))
    const better = [{ id: 'r01', index: 9, name: 'red', value: 40000, can_protect: true }]
    const marginal = [{ id: 'g01', index: 9, name: 'gold', value: 11000, can_protect: true }]
    const d = makePolicy().decide(makeState({ bag: better, safe, controls: ['rush'] }))
    expect(d.command).toBe('swap_box')
    expect(d.params.index).toBe(9)
    expect(makePolicy().decide(makeState({ bag: marginal, safe, controls: ['rush'] })).command).toBe('rush')
})

it('只剩战地手术匣时：楼层间单步，遇敌时使用，有其他回复则保留', () => {
    const kit = [{ id: 'field_surgery', index: 4, name: '战地手术匣' }]
    expect(makePolicy().decide(makeState({ hp: 250, bag: kit, controls: ['rush', 'down', 'extract'] })).command).toBe(
        'down'
    )
    const mob = { name: 'mob', hp: 50, atk: 10, def: 2, agi: 5 }
    const d = makePolicy().decide(makeState({ hp: 180, enemy: mob, bag: kit }))
    expect(d.command).toBe('use')
    expect(d.params).toEqual({ index: 4 })
    // 高于使用阈值时，简单战斗直接打，不花手术匣
    expect(makePolicy().decide(makeState({ hp: 250, enemy: mob, bag: kit })).command).toBe('fight')
    const both = [...kit, { id: 'coffee', index: 5, name: '冰美式' }]
    expect(makePolicy().decide(makeState({ hp: 250, enemy: mob, bag: both })).command).toBe('fight')
})

it('连战判断忽略会失效的限层增益', () => {
    const policy = makePolicy()
    const state = makeState({ hp: 463, bag: KIT, controls: ['rush', 'down', 'extract'] })
    state.run.floor = 1162
    state.run.effective_stats = { hp: 670, atk: 276, def: 112, agi: 71, luk: 21 }
    state.run.field_buffs = []
    policy.decide(state) // 记住无增益时的属性
    state.run.effective_stats = { hp: 670, atk: 276, def: 147, agi: 71, luk: 21 }
    state.run.field_buffs = [{ id: 'field_plate' }] // 临时防御增益生效中
    expect(policy.lastingBuild(state).defense).toBe(112)
    expect(policy.decide(state).command).toBe('down') // F1164 阵亡的情形
})

it('回复手段用完时撤离', () => {
    const state = makeState({ hp: 600, controls: ['rush', 'down', 'extract'] })
    state.run.floor = 900
    expect(makePolicy().decide(state).command).toBe('extract')
})

it('没有回复时遇到高风险战斗则撤离', () => {
    const elite = { name: 'elite', elite: true, hp: 2700, atk: 300, def: 90, agi: 55 }
    const d = makePolicy().decide(makeState({ hp: 300, enemy: elite }))
    expect(d.command, d.reason).toBe('extract')
})

it('还有回复时继续前进', () => {
    const state = makeState({
        hp: 600,
        bag: [{ id: 'coffee', index: 2, name: '冰美式' }],
        controls: ['rush', 'down', 'extract']
    })
    state.run.floor = 40
    expect(makePolicy().decide(state).command).toBe('rush')
})

it('背包里有替身娃娃时不撤离', () => {
    const doll = [{ id: 'doll', index: 7, name: '替身娃娃' }]
    const state = makeState({ hp: 600, bag: doll, controls: ['rush', 'down', 'extract'] })
    state.run.floor = 900
    expect(makePolicy().decide(state).command).not.toBe('extract') // 没有回复，但有娃娃：继续
    const elite = { name: 'elite', elite: true, hp: 2700, atk: 300, def: 90, agi: 55 }
    expect(makePolicy().decide(makeState({ hp: 300, enemy: elite, bag: doll })).command).toBe('fight')
})

it('默认不使用扩容袋', () => {
    const state = makeState({ hp: 666, bag: [...KIT, { id: 'pouch', index: 3, name: '扩容袋' }], controls: ['rush'] })
    state.run.floor = 40
    expect(makePolicy().decide(state).command).toBe('rush')
    expect(makePolicy({ usePouch: true }).decide(state).params).toEqual({ index: 3 })
})

describe('临装选择', () => {
    const offer = (description: string) => ({
        name: '临装',
        description,
        choices: [0, 1, 2].map(slot => ({ control: { params: { id: 'x', slot } } }))
    })

    it('空位直接佩戴，负分放弃', () => {
        const state = makeState({ controls: ['field_pick'] })
        state.run.field_offer = offer('本局佩戴 · 闪避率+11% · 攻击+13%')
        expect(makePolicy().decide(state).params).toEqual({ id: 'x', slot: 1 })
        state.run.field_offer = offer('本局佩戴 · 攻击-20%')
        expect(makePolicy().decide(state).params).toEqual({ id: 'x', slot: 0 })
    })

    it('「伤害 ×2」默认不戴', () => {
        const state = makeState({ controls: ['field_pick'] })
        state.run.field_offer = offer('策划的平衡性调整 · 造成与承受伤害 ×2')
        expect(makePolicy().decide(state).params.slot).toBe(0)
        expect(makePolicy({ balanceGearScore: 30 }).decide(state).params.slot).toBe(1)
    })

    it('Python 中为空的 {} 与 null 一样不触发临装逻辑', () => {
        const state = makeState({ hp: 666, controls: ['rush'] })
        state.run.field_offer = {}
        state.run.floor = 40
        expect(makePolicy().decide(state).command).toBe('rush')
    })
})
