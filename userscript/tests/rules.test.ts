// ============================================================================
// rules.test.ts — 2026-11-25 服务器更新后的规则与 Python 版对齐的部分
// （深层破甲、遭遇概率、首领墙、祝福与虹彩效果、遗物盒、道具拒绝）
// ============================================================================

import { describe, expect, it } from 'vitest'
import { battleEnemy, Calibration, rainbowFleeBonus, runBuild } from '../src/combat.ts'
import { buildEffects } from '../src/heirlooms.ts'
import { bossWinChance, depthShred, fight, floorMix, makeBuild, nextBossFloors, parseFloorOdds } from '../src/model.ts'
import { BossFarmPolicy, relicValue } from '../src/policy.ts'
import { seededRng } from '../src/rng.ts'
import type { Json } from '../src/types.ts'

const BUILD = makeBuild({ hp: 666, atk: 120, defense: 20, agi: 30, luk: 21, pierce: 0.26, leech: 0.06, critMult: 1.9 })

function makePolicy(): BossFarmPolicy {
    const policy = new BossFarmPolicy(new Calibration({ k_player: 1.0, k_enemy: 1.0 }))
    policy.affixBuild = BUILD
    policy.loadoutKey = JSON.stringify([])
    return policy
}

describe('模型', () => {
    it('深层破甲与实战值一致', () => {
        expect(depthShred(50, 0.6)).toBe(0)
        expect(depthShred(51, 0.6)).toBeCloseTo(0.0069767441860465115, 12)
        expect(depthShred(63, 0.6)).toBeCloseTo(0.07959183673469387, 12)
    })

    it('遭遇概率与页面显示一致，遗物计入宝箱', () => {
        const shown = parseFloorOdds('📊 F247：箱34% 怪41% 精12% 奇13%｜金红68.0%(红11.9%)')
        expect(shown).toEqual({ normal: 0.41, elite: 0.12, choice: 0.13, loot: 0.34 })
        const model = floorMix(247)
        for (const k of ['normal', 'elite', 'choice', 'loot'] as const) expect(model[k]).toBeCloseTo(shown![k], 5)
        expect(parseFloorOdds('📊 F250 必遇首领。')).toBeNull()
        expect(parseFloorOdds('📊 F300：箱30% 怪33% 精20% 奇13% 遗4%｜')!.loot).toBeCloseTo(0.34, 9)
    })

    it('首领墙：打不过的首领胜率为 0，弱首领接近 1', () => {
        const rng = seededRng(3)
        expect(bossWinChance(BUILD, { hp: 300, atk: 30, def: 5, agi: 10 }, { samples: 200, rng })).toBeGreaterThan(0.99)
        expect(bossWinChance(BUILD, { hp: 1e6, atk: 5000, def: 500, agi: 55 }, { samples: 200, rng })).toBe(0)
        expect(nextBossFloors(28, 3)).toEqual([30, 50, 70])
    })

    it('抢拍回合承受伤害 ×0.65', () => {
        const foe = { hp: 1e6, atk: 100, def: 0, agi: 0 }
        const base = makeBuild({ hp: 1000, atk: 1, defense: 0, agi: 0, luk: 0 })
        const taken = (beat: boolean) => {
            const res = fight({ ...base, beat }, new Float64Array(4000).fill(1000), new Float64Array(4000), foe, {
                boss: false,
                rounds: 4,
                rng: seededRng(0)
            })
            return 1000 - res.hp.reduce((a, b) => a + b, 0) / 4000
        }
        expect(taken(true) / taken(false)).toBeCloseTo((3 + 0.65) / 4, 1)
    })
})

describe('实时 Build', () => {
    it('缺刻值夜表、深层破甲上限、风隙羽标与破甲后的敌防', () => {
        expect(buildEffects({ slots: ['B06-1', 'B06-2', 'B06-3', 'B06-4'], pieces: {} })).toEqual({
            pierceCap: 0.87,
            beat: true
        })
        const state: Json = {
            run: { max_hp: 702, effective_stats: { atk: 228, def: 122, agi: 79, luk: 24 } },
            player: { secondary: { pierce: 0.38, condition: { depth_cap: 0.6, depth_shred: 0.25 } } },
            heirlooms: { slots: ['B06-1', 'B06-2', 'B06-3', 'B06-4'], pieces: {} },
            atlas: { rainbow: [{ id: 'prism_wind', unlocked: true }] }
        }
        const b = runBuild(state, BUILD)
        expect([b.beat, b.pierceCap, b.depthShredCap, b.fleeBonus]).toEqual([true, 0.87, 0.6, 0.2])
        expect(rainbowFleeBonus(state)).toBeCloseTo(0.2, 9)
        expect(battleEnemy(state, { hp: 5, atk: 1, def: 80, agi: 0 }).def).toBe(60)
    })
})

describe('遗物盒与道具拒绝', () => {
    const relicState = (value: number): Json => ({
        run: {
            floor: 900,
            hp: 700,
            max_hp: 700,
            bag: [{ id: 'coffee', index: 9, name: 'coffee', quantity: 5, value: 300, can_protect: true }],
            safe: [],
            safe_capacity: 4,
            event: { kind: 'relic', name: '遗物盒', text: `某人 遗落的物资，估值 ${value.toLocaleString('en')} 金。` },
            effective_stats: { hp: 700, atk: 150, def: 70, agi: 77, luk: 21 }
        },
        player: { loadout: {}, secondary: {} },
        controls: ['rob', 'rescue', 'leave', 'extract'].map(c => ({ command: c, params: {}, label: c })),
        atlas: { items: [] }
    })

    it('估值 ≥ 1000 万舔包，否则捞尸', () => {
        expect(relicValue({ text: '某人 遗落的物资，估值 12,345,678 金。' })).toBe(12345678)
        expect(makePolicy().decide(relicState(15_000_000)).command).toBe('rob')
        expect(makePolicy().decide(relicState(4_000_000)).command).toBe('rescue')
    })

    it('舔包后放不进安全箱的部分 ≥ 1000 万时撤离；普通战利品不触发', () => {
        const policy = makePolicy()
        const loot = Array.from({ length: 5 }, (_, i) => ({
            id: `g${i}`,
            index: i + 1,
            name: `g${i}`,
            value: 3_000_000,
            can_protect: true
        }))
        const safe = Array.from({ length: 4 }, (_, k) => ({ id: 's', index: k + 1, name: 's', value: 6_000_000 }))
        const s = relicState(0)
        s.run = { ...s.run, bag: [...loot, ...s.run.bag], safe, event: { kind: 'ready' } }
        s.controls = ['down', 'rush', 'extract'].map(c => ({ command: c, params: {}, label: c }))
        policy.newRun()
        policy.banking = true
        expect(policy.decide(s).command).toBe('extract')
    })

    it('肾上腺素因攻击加成上限被拒绝后本局不再尝试', () => {
        const policy = makePolicy()
        policy.ban(
            616,
            { command: 'use', params: { index: 26 }, reason: '' },
            'adrenaline',
            '肾上腺素最多把本局攻击加成叠到 +50%'
        )
        expect(policy.adrenalineUsed).toBe(policy.cfg.startAdrenaline)
    })
})
