// ============================================================================
// model.ts — 离线游戏模型：怪物成长公式 + 战斗蒙特卡洛（移植自 model.py）
//
// 规则来自游戏内公式表（docs/formulas_zh.txt），怪物数值已与实战数据核对
// （如商场 F1030 首领：预测与实测均为 4,067 HP）。
// Python 版用 numpy 对 N 个样本做向量化运算；样本之间互不影响，这里逐样本循环，语义相同。
// ============================================================================

import type { Rng } from './rng.ts'

export const MAP_MULT: Record<string, number> = {
    mall: 1.0,
    subway: 1.6,
    abyss: 2.45,
    auction: 3.3,
    tide: 3.85,
    zenith: 4.45
}

export type EnemyKind = 'normal' | 'elite' | 'boss'
type Growth = Record<'hp' | 'atk' | 'def' | 'agi', [number, number, number]>

/** 每项属性 = a + b*x + c*z */
const GROWTH: Record<EnemyKind, Growth> = {
    normal: { hp: [40, 1.15, 4.5], atk: [10.8, 0.13, 0.88], def: [2.7, 0.045, 0.16], agi: [7, 0.09, 0.58] },
    elite: { hp: [74, 1.9, 11.4], atk: [15.2, 0.16, 1.16], def: [3.1, 0.07, 0.24], agi: [8, 0.11, 0.74] },
    boss: { hp: [198, 2.6, 7.1], atk: [30.5, 0.21, 0.58], def: [4.4, 0.095, 0.33], agi: [9, 0.12, 0.8] }
}

export const BOSS_WINDOW = 4
export const BOSS_HEAVY = 1.26
/** 普通怪 15 回合未分胜负即撤退（无战利品） */
export const REGULAR_ROUND_CAP = 15
/** 精英（2026-11-25 起）：最多 16 回合，第 8 回合后暂停整备（可用药、换临装），平局仍撤退 */
export const ELITE_ROUND_CAP = 16
export const ELITE_BREAK = 8

export interface Foe {
    hp: number
    atk: number
    def: number
    agi: number
    [key: string]: unknown
}

/** 现在开始的这一段战斗：[起始回合, 回合数]，到首领窗口结束、精英整备或回合上限为止 */
export function fightSegment(foe: Foe): [number, number] {
    const fought = Number(foe.rounds_fought || 0)
    if (foe.boss) return [fought + 1, BOSS_WINDOW - (fought % BOSS_WINDOW)]
    const end = foe.elite ? (fought < ELITE_BREAK ? ELITE_BREAK : ELITE_ROUND_CAP) : REGULAR_ROUND_CAP
    return [fought + 1, Math.max(1, end - fought)]
}

export function bossIndex(floor: number): number {
    const fixed: Record<number, number> = { 8: 1, 18: 2, 30: 3 }
    return fixed[floor] || 3 + Math.floor((floor - 30) / 20)
}

export function isBossFloor(floor: number): boolean {
    return floor === 8 || floor === 18 || floor === 30 || (floor > 30 && (floor - 30) % 20 === 0)
}

const enemyCache = new Map<string, Foe>()

export function enemy(kind: EnemyKind, floor: number, mapId = 'mall'): Foe {
    const key = `${kind}|${floor}|${mapId}`
    const cached = enemyCache.get(key)
    if (cached) return { ...cached }
    const x = floor - 1
    const z = 6 * Math.log(1 + x / 6)
    const g = GROWTH[kind]
    const stat = (k: keyof Growth) => g[k][0] + g[k][1] * x + g[k][2] * z
    const m = MAP_MULT[mapId]
    if (m === undefined) throw new Error(`未知地图 ${mapId}`)
    let hp = stat('hp') * m
    let atk = stat('atk') * m
    const def = stat('def') * m
    const agi = Math.min(55.0, stat('agi'))
    if (kind === 'boss') {
        const n = bossIndex(floor)
        hp *= 1 + 0.055 * Math.log2(n)
        atk *= 1 + 0.033 * Math.log2(n)
    }
    const out: Foe = { hp: Math.floor(hp), atk: Math.floor(atk), def: Math.floor(def), agi: Math.floor(agi) }
    enemyCache.set(key, out)
    return { ...out }
}

/** 战斗相关的全部属性。比率均为小数（0.26 = 26%）。 */
export interface Build {
    hp: number
    atk: number
    defense: number
    agi: number
    luk: number
    pierce: number
    leech: number
    combo: number
    /** 重击：首回合主攻加成 */
    heavy: number
    /** 越战越勇：每回合攻击加成（上限 75%） */
    berserk: number
    /** 嗜血：击杀回复最大生命的比例 */
    vampire: number
    /** 反甲刺猬：反伤比例（上限 70%） */
    thorns: number
    /** 对首领伤害（上限 120%） */
    bossBonus: number
    critMult: number
    /** 临装暴击率修正 */
    critBonus: number
    /** 孤注一掷 */
    desperate: boolean
    /** 不屈 */
    lastStand: boolean
    startShield: number
    /** 临装双向倍率（策划的平衡性调整：造成 ×2 / 承受 ×2） */
    dealtMult: number
    takenMult: number
    /** 减伤（负数 = 承受更多伤害） */
    reduction: number
    defPct: number
    atkPct: number
    label: string
}

export type BuildInput = Partial<Build> & Pick<Build, 'hp' | 'atk' | 'defense' | 'agi' | 'luk'>

export function makeBuild(input: BuildInput): Build {
    return {
        pierce: 0,
        leech: 0,
        combo: 0,
        heavy: 0,
        berserk: 0,
        vampire: 0,
        thorns: 0,
        bossBonus: 0,
        critMult: 1.6,
        critBonus: 0,
        desperate: false,
        lastStand: false,
        startShield: 0,
        dealtMult: 1,
        takenMult: 1,
        reduction: 0,
        defPct: 0,
        atkPct: 0,
        label: '',
        ...input
    }
}

export const effDef = (b: Build): number => Math.max(0, b.defense * (1 + b.defPct))
export const effAtk = (b: Build): number => b.atk * Math.max(0.25, 1 + b.atkPct)
export const dodge = (b: Build): number => Math.min(0.52, Math.max(0, Math.min(0.45, 0.03 + 0.004 * b.agi)))

export function crit(b: Build, lowHp = false): number {
    const base = Math.min(0.6, (0.05 + 0.005 * Math.max(0, b.luk)) * (lowHp && b.desperate ? 2 : 1))
    return Math.min(0.65, Math.max(0, base + b.critBonus))
}

/** 商场临装池中可佩戴的临装，作为 Build 修正 */
export const TEMP_GEAR: Record<string, (b: Build) => Partial<Build>> = {
    // 策划的平衡性调整
    balance: b => ({ dealtMult: b.dealtMult * 2, takenMult: b.takenMult * 2 }),
    // 厚帆布围裙
    apron: b => ({ defPct: b.defPct + 0.3, agi: b.agi - 3 }),
    // 通勤束带
    harness: b => ({ defPct: b.defPct + 0.15, agi: b.agi + 5 }),
    // 促销聚光灯
    spotlight: b => ({ critBonus: b.critBonus + 0.13, reduction: b.reduction - 0.1 })
}

export function withTempGear(build: Build, ...items: string[]): Build {
    return items.reduce((b, item) => ({ ...b, ...TEMP_GEAR[item](b) }), build)
}

export interface FightOptions {
    boss: boolean
    ehp?: Float64Array
    startRound?: number
    rounds?: number
    lastStand?: Uint8Array
    rng: Rng
}

export interface FightResult {
    hp: Float64Array
    shield: Float64Array
    ehp: Float64Array
    lastStand: Uint8Array
    used: Float64Array
}

/**
 * N 个并行样本的战斗。首领战传 rounds = 4 - ((startRound - 1) % 4)，在窗口末尾停下。
 * hp <= 0 的样本已阵亡；ehp <= 0 表示敌人已死（击杀回复已计入）。
 */
export function fight(
    build: Build,
    hp0: ArrayLike<number>,
    shield0: ArrayLike<number>,
    foe: Foe,
    opts: FightOptions
): FightResult {
    const { boss, rng } = opts
    const startRound = opts.startRound ?? 1
    const rounds = opts.rounds ?? 80
    const n = hp0.length
    const hp = Float64Array.from(hp0)
    const shield = Float64Array.from(shield0)
    const ehp = opts.ehp ? Float64Array.from(opts.ehp) : new Float64Array(n).fill(foe.hp)
    const ls = opts.lastStand ? Uint8Array.from(opts.lastStand) : new Uint8Array(n).fill(build.lastStand ? 1 : 0)
    const used = new Float64Array(n)

    const maxHp = build.hp
    const edodge = Math.min(0.45, 0.03 + 0.004 * foe.agi)
    const pdodge = dodge(build)
    const pRaw = (effAtk(build) * 100) / (100 + 2.5 * foe.def * (1 - Math.min(0.75, build.pierce)))
    const eRaw = ((foe.atk * 100) / (100 + 2.5 * effDef(build))) * (1 - build.reduction) * build.takenMult
    const bossMult = boss ? 1 + Math.min(1.2, build.bossBonus) : 1.0
    const playerFirst = build.agi >= foe.agi
    const critLow = crit(build, true)
    const critHigh = crit(build, false)
    const comboP = Math.min(0.7, build.combo)
    const leechF = Math.min(0.42, build.leech)
    const leechCap = Math.max(1.0, 0.1 * maxHp)
    const thornsF = Math.min(0.7, build.thorns)
    const uniform = () => 0.9 + 0.2 * rng.random()

    for (let s = 0; s < n; s++) {
        let h = hp[s]
        let sh = shield[s]
        let e = ehp[s]
        let l = ls[s]

        const playerAttack = (t: number) => {
            const low = h < 0.3 * maxHp
            let mult = (1 + Math.min(0.75, (t - 1) * build.berserk)) * bossMult
            if (t === 1) mult *= 1 + build.heavy
            const atkMult = low && build.desperate ? 1.75 : 1.0
            const hit = rng.random() >= edodge
            const isCrit = rng.random() < (low ? critLow : critHigh)
            let dmg = pRaw * mult * atkMult * uniform() * (isCrit ? build.critMult : 1.0)
            dmg = hit ? Math.floor(dmg * build.dealtMult) : 0
            if (build.combo > 0) {
                const extra = rng.random() < comboP && rng.random() >= edodge
                const bonus = Math.floor(0.6 * pRaw * mult * atkMult * uniform() * build.dealtMult)
                if (extra) dmg += bonus
            }
            dmg = Math.max(dmg, hit ? 1 : 0)
            const dealt = Math.min(dmg, Math.max(e, 0))
            e -= dmg
            if (build.leech > 0) {
                const heal = Math.min(Math.floor(dealt * leechF), leechCap)
                h = Math.min(maxHp, h + heal)
            }
        }

        const enemyAttack = (t: number) => {
            const hit = rng.random() >= pdodge
            let dmg = eRaw * uniform()
            if (boss && t % BOSS_WINDOW === 0) dmg *= BOSS_HEAVY
            dmg = hit ? Math.floor(Math.max(dmg, 1)) : 0
            const absorbed = Math.min(sh, dmg)
            sh -= absorbed
            h -= dmg - absorbed
            if (build.thorns > 0 && h > 0) e -= Math.floor(dmg * thornsF)
            if (l && h <= 0) {
                h = 1.0
                l = 0
            }
        }

        for (let i = 0; i < rounds; i++) {
            if (!(h > 0 && e > 0)) break
            const t = startRound + i
            used[s] += 1
            if (playerFirst) {
                playerAttack(t)
                if (h > 0 && e > 0) enemyAttack(t)
            } else {
                enemyAttack(t)
                if (h > 0 && e > 0) playerAttack(t)
            }
        }
        if (build.vampire > 0 && e <= 0 && h > 0) h = Math.min(maxHp, h + build.vampire * maxHp)
        hp[s] = h
        shield[s] = sh
        ehp[s] = e
        ls[s] = l
    }
    return { hp, shield, ehp, lastStand: ls, used }
}
