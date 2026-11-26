// ============================================================================
// parity.test.ts — 与 Python 版（elevator_agent）对照
//
// tests/fixtures/parity.json 由 scripts/gen_parity.py 用 Python 版生成：
// 确定性部分（怪物成长、评分、装备、校准）要求完全一致；
// 蒙特卡洛部分（战斗 / 连战风险、决策）按统计误差比较。
// ============================================================================

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { Calibration, chainRisk, parseBattleLog, runBuild } from '../src/combat.ts'
import { GearModel } from '../src/gear.ts'
import { enemy, fight, makeBuild } from '../src/model.ts'
import type { Build, EnemyKind } from '../src/model.ts'
import { BossFarmPolicy, DEFAULT_POLICY } from '../src/policy.ts'
import { seededRng } from '../src/rng.ts'
import type { Json } from '../src/types.ts'

const fx: Json = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf-8'))
const AFFIXES: Build = makeBuild(fx.affixes)

/** 两个独立估计之差的容差：5 倍标准误 + 0.2 个百分点 */
function tolerance(p: number, nA: number, nB: number): number {
    const v = Math.max(p * (1 - p), 1e-4)
    return 5 * Math.sqrt(v / nA + v / nB) + 0.002
}

describe('怪物成长公式', () => {
    it('所有地图、楼层、怪物类型与 Python 完全一致', () => {
        const wrong = fx.enemies.filter((e: Json) => {
            const got = enemy(e.kind as EnemyKind, e.floor, e.map)
            return got.hp !== e.foe.hp || got.atk !== e.foe.atk || got.def !== e.foe.def || got.agi !== e.foe.agi
        })
        expect(fx.enemies.length).toBeGreaterThan(4000)
        expect(wrong).toEqual([])
    })
})

describe('战斗引擎', () => {
    for (const c of fx.fights) {
        it(`${c.name} vs ${c.boss ? '首领' : '怪'} hp=${c.foe.hp} 起始回合 ${c.startRound}`, () => {
            const n = 40_000
            const res = fight(
                makeBuild(c.build),
                new Float64Array(n).fill(c.hp),
                new Float64Array(n).fill(c.shield),
                c.foe,
                {
                    boss: c.boss,
                    startRound: c.startRound,
                    rounds: c.rounds,
                    rng: seededRng(7 + c.foe.hp)
                }
            )
            let dead = 0
            let won = 0
            let hpSum = 0
            for (let i = 0; i < n; i++) {
                if (res.hp[i] <= 0) dead++
                else {
                    hpSum += res.hp[i]
                    if (res.ehp[i] <= 0) won++
                }
            }
            expect(Math.abs(dead / n - c.death)).toBeLessThanOrEqual(tolerance(c.death, n, c.samples))
            expect(Math.abs(won / n - c.win)).toBeLessThanOrEqual(tolerance(c.win, n, c.samples))
            if (n - dead > 1000 && c.hpAfter > 0) {
                expect(Math.abs(hpSum / (n - dead) / c.hpAfter - 1)).toBeLessThan(0.02)
            }
        })
    }
})

describe('连战风险', () => {
    for (const c of fx.chains) {
        it(`F${c.state.run.floor} HP ${c.state.run.hp}${c.elites ? '（含精英）' : ''}`, () => {
            const cal = new Calibration()
            const risk = chainRisk(
                runBuild(c.state, AFFIXES),
                c.state,
                cal,
                4,
                40_000,
                'mall',
                seededRng(c.state.run.floor),
                Boolean(c.elites)
            )
            expect(Math.abs(risk - c.risk)).toBeLessThanOrEqual(tolerance(c.risk, 40_000, 40_000))
        })
    }
})

describe('决策（练习局轨迹中的真实状态）', () => {
    it('与 Python 一致；不一致只允许出现在风险阈值附近', () => {
        const mismatches: string[] = []
        const unexplained: string[] = []
        const cal = new Calibration()
        for (const d of fx.decisions) {
            const base = fx.states[d.index]
            const state = {
                run: { ...base.run, ...d.patch },
                controls: base.controls,
                player: fx.player,
                atlas: { items: [] }
            }
            const policy = new BossFarmPolicy(cal, { samples: 4000 })
            policy.affixBuild = AFFIXES
            policy.loadoutKey = JSON.stringify([])
            const got = policy.decide(state)
            if (got.command === d.command && JSON.stringify(got.params) === JSON.stringify(d.params)) continue
            const label = `#${d.index}/${d.variant} F${state.run.floor}: Python ${d.command} ${JSON.stringify(d.params)}，JS ${got.command} ${JSON.stringify(got.params)}`
            mismatches.push(label)
            const boss = Boolean(state.run.event?.enemy?.boss)
            const limit = boss ? DEFAULT_POLICY.maxBossWindowRisk : DEFAULT_POLICY.maxDeathRisk
            const near =
                (d.death !== undefined &&
                    (Math.abs(d.death - limit) < 0.012 || Math.abs(d.death - DEFAULT_POLICY.abortRisk) < 0.03)) ||
                (d.chainRisk !== undefined && Math.abs(d.chainRisk - DEFAULT_POLICY.maxRushRisk) < 0.006)
            if (!near) unexplained.push(`${label}（death=${d.death} chain=${d.chainRisk}）`)
        }
        if (mismatches.length) console.info(`阈值附近的随机差异 ${mismatches.length} 处：\n${mismatches.join('\n')}`)
        expect(fx.decisions.length).toBeGreaterThan(200)
        expect(unexplained).toEqual([])
        expect(mismatches.length).toBeLessThan(fx.decisions.length * 0.05)
    })
})

describe('评分', () => {
    it('奇遇选项评分与 Python 完全一致', () => {
        for (const e of fx.scores.events) {
            expect(BossFarmPolicy.scoreOption(e.desc, e.hpFrac, e.maxHp, e.capped), e.desc).toBeCloseTo(e.score, 9)
        }
        expect(fx.scores.events.length).toBeGreaterThan(10)
    })

    it('临装评分与 Python 完全一致', () => {
        for (const w of fx.scores.wears) expect(BossFarmPolicy.wearScore(w.desc), w.desc).toBeCloseTo(w.score, 9)
    })
})

describe('校准', () => {
    it('逐场学习的系数与 Python 完全一致', () => {
        const cal = new Calibration()
        for (const step of fx.calibration.steps) {
            expect(parseBattleLog(step.log)).toEqual(step.hits)
            cal.observe(makeBuild(step.build), step.enemy, step.log)
        }
        const f = fx.calibration.final
        expect(cal.nPlayer).toBe(f.n_player)
        expect(cal.nEnemy).toBe(f.n_enemy)
        expect(cal.kPlayer).toBeCloseTo(f.k_player, 12)
        expect(cal.kEnemy).toBeCloseTo(f.k_enemy, 12)
        expect(fx.calibration.steps.length).toBeGreaterThan(5)
    })
})

describe('装备 → Build', () => {
    const gm = new GearModel(fx.gear.state)

    it('基础属性与套装加成解析一致', () => {
        expect(gm.base).toEqual(fx.gear.base)
        // Python 字段名为 snake_case，JS 为 camelCase
        const camel = (k: string) => k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())
        const want: Record<string, Record<string, Record<string, number>>> = {}
        for (const [name, tiers] of Object.entries(fx.gear.sets) as [string, Json][]) {
            want[name] = {}
            for (const [p, bonus] of Object.entries(tiers) as [string, Record<string, number>][]) {
                want[name][p] = Object.fromEntries(Object.entries(bonus).map(([k, v]) => [camel(k), v]))
            }
        }
        expect(JSON.parse(JSON.stringify(gm.sets))).toEqual(want)
    })

    it('当前穿戴与混搭组合的 Build 一致', () => {
        const byId = new Map(gm.items.map(i => [i.equipId, i]))
        const builds = {
            current: gm.build(gm.current),
            mixed: gm.build([5, 6, 3, 4].map(i => byId.get(i)!))
        }
        for (const [name, want] of Object.entries(fx.gear.builds) as [string, Json][]) {
            const got = builds[name as keyof typeof builds] as unknown as Record<string, unknown>
            for (const [k, v] of Object.entries(want)) {
                if (typeof v === 'number') expect(got[k], `${name}.${k}`).toBeCloseTo(v, 9)
                else expect(got[k], `${name}.${k}`).toEqual(v)
            }
        }
    })
})
