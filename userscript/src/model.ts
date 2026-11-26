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
/** 缺刻值夜表四件（抢拍）：对同一敌人的第 4、8、12……回合先手，承受伤害 ×0.65 */
export const BEAT_EVERY = 4
export const BEAT_TAKEN = 0.65
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
    /** 局内道具闪避修正（玩家闪避上限 52%） */
    dodgeBonus: number
    /** 防护类局内道具（防碎封箱带等）：接下来 N 次被命中减伤 */
    guardHits: number
    guardReduction: number
    /** 蓄力类局内道具（热敏收银纸等）：接下来 N 次主攻加成 */
    burstHits: number
    burstMult: number
    /** 蓄力主攻必中 */
    burstSure: boolean
    /** 穿透上限：缺刻值夜表两件后 87% */
    pierceCap: number
    /** 缺刻值夜表四件（抢拍） */
    beat: boolean
    /** 闪耀祝福深层破甲上限（见 depthShred） */
    depthShredCap: number
    /** 风隙羽标：逃跑率在常规上限后再加 20 个百分点（最高 100%） */
    fleeBonus: number
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
        dodgeBonus: 0,
        guardHits: 0,
        guardReduction: 0,
        burstHits: 0,
        burstMult: 1,
        burstSure: false,
        pierceCap: 0.75,
        beat: false,
        depthShredCap: 0,
        fleeBonus: 0,
        label: '',
        ...input
    }
}

export const effDef = (b: Build): number => Math.max(0, b.defense * (1 + b.defPct))
export const effAtk = (b: Build): number => b.atk * Math.max(0.25, 1 + b.atkPct)
export const dodge = (b: Build): number =>
    Math.min(0.52, Math.max(0, Math.min(0.45, 0.03 + 0.004 * b.agi) + b.dodgeBonus))

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
 * N 个并行样本的战斗（与 model.py 的 fight 一致，逐样本循环）。首领战传 rounds = 4 - ((startRound - 1) % 4)，
 * 在窗口末尾停下。hp <= 0 的样本已阵亡；ehp <= 0 表示敌人已死（击杀回复已计入）。
 * 按服务器顺序：首回合加成与蓄力只作用于主攻；追击只在敌人仍存活时发生；防护次数按实际被命中消耗。
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
    const rounds = opts.rounds ?? REGULAR_ROUND_CAP
    const n = hp0.length
    const hp = Float64Array.from(hp0)
    const shield = Float64Array.from(shield0)
    const ehp = opts.ehp ? Float64Array.from(opts.ehp) : new Float64Array(n).fill(foe.hp)
    const ls = opts.lastStand ? Uint8Array.from(opts.lastStand) : new Uint8Array(n).fill(build.lastStand ? 1 : 0)
    const used = new Float64Array(n)

    const maxHp = build.hp
    const edodge = Math.min(0.45, 0.03 + 0.004 * foe.agi)
    const pdodge = dodge(build)
    const pRaw = (effAtk(build) * 100) / (100 + 2.5 * foe.def * (1 - Math.min(build.pierceCap, build.pierce)))
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
        let guardLeft = build.guardHits
        let burstLeft = build.burstHits

        /** 单次命中：取整 → 暴击 → 双向倍率 */
        const strike = (mult: number, atkMult: number, critP: number): number => {
            let dmg = Math.max(1, Math.floor(pRaw * mult * atkMult * uniform()))
            if (rng.random() < critP) dmg = Math.floor(dmg * build.critMult)
            return Math.max(1, Math.floor(dmg * build.dealtMult))
        }

        const playerAttack = (t: number) => {
            const low = h < 0.3 * maxHp
            const mult = (1 + Math.min(0.75, (t - 1) * build.berserk)) * bossMult
            const atkMult = low && build.desperate ? 1.75 : 1.0
            const critP = low ? critLow : critHigh
            const charged = burstLeft > 0
            if (charged) burstLeft -= 1
            const sure = charged && build.burstSure
            const main = mult * (t === 1 ? 1 + build.heavy : 1.0) * (charged ? build.burstMult : 1.0)
            const hit = rng.random() >= edodge || sure
            let dmg = strike(main, atkMult, critP) * (hit ? 1 : 0)
            if (build.combo > 0) {
                const extra = e > dmg && rng.random() < comboP && rng.random() >= edodge
                dmg += strike(0.6 * mult, atkMult, critP) * (extra ? 1 : 0)
            }
            const dealt = Math.min(dmg, Math.max(e, 0))
            e -= dmg
            if (build.leech > 0) h = Math.min(maxHp, h + Math.min(Math.floor(dealt * leechF), leechCap))
        }

        const enemyAttack = (t: number, beatRound: boolean) => {
            const hit = rng.random() >= pdodge
            let dmg = eRaw * uniform()
            if (boss && t % BOSS_WINDOW === 0) dmg *= BOSS_HEAVY
            if (beatRound) dmg *= BEAT_TAKEN
            if (hit && guardLeft > 0) {
                dmg *= 1 - build.guardReduction
                guardLeft -= 1
            }
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
            const beatRound = build.beat && t % BEAT_EVERY === 0
            used[s] += 1
            if (playerFirst || beatRound) {
                playerAttack(t)
                if (h > 0 && e > 0) enemyAttack(t, beatRound)
            } else {
                enemyAttack(t, beatRound)
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

// ---------------------------------------------------------------------------
// 2026-11-25 起的服务器规则（与 model.py 一致）
// ---------------------------------------------------------------------------

const DEPTH_SHRED_FROM = 50
const DEPTH_SHRED_HALF = 85

/**
 * 闪耀祝福深层破甲（本层祝福深层破甲）：上限 × d / (d + 85)，d = 楼层 − 50。
 * 与实战 secondary.condition.depth_shred 完全吻合；与穿透相乘：有效敌防 = 敌防 × (1 − 破甲) × (1 − 穿透)。
 */
export function depthShred(floor: number, cap: number): number {
    const d = Math.max(0, floor - DEPTH_SHRED_FROM)
    return (cap * d) / (d + DEPTH_SHRED_HALF)
}

/** 按此 Build 的深层破甲削减防御后的怪物（伤害公式看到的样子） */
export function foeAt(kind: EnemyKind, floor: number, mapId: string, build: Build): Foe {
    const foe = enemy(kind, floor, mapId)
    const shred = depthShred(floor, build.depthShredCap)
    return shred ? { ...foe, def: foe.def * (1 - shred) } : foe
}

/**
 * 每个整备窗口前都回满、回复永不耗尽时击杀首领的概率：乐观上界，用来区分打得过的首领和打不过的墙。
 * foe.hp 为满血，ehp 为剩余血量。
 */
export function bossWinChance(
    build: Build,
    foe: Foe,
    opts: { ehp?: number; startRound?: number; samples?: number; maxWindows?: number; rng: Rng }
): number {
    const samples = opts.samples ?? 600
    const left = new Float64Array(samples).fill(opts.ehp ?? foe.hp)
    const alive = new Uint8Array(samples).fill(1)
    let first = opts.startRound ?? 1
    for (let w = 0; w < (opts.maxWindows ?? 80); w++) {
        let going = 0
        for (let s = 0; s < samples; s++) if (alive[s] && left[s] > 0) going++
        if (!going) break
        const rounds = BOSS_WINDOW - ((first - 1) % BOSS_WINDOW)
        const res = fight(build, new Float64Array(samples).fill(build.hp), new Float64Array(samples), foe, {
            boss: true,
            ehp: left,
            startRound: first,
            rounds,
            rng: opts.rng
        })
        for (let s = 0; s < samples; s++) {
            if (!(alive[s] && left[s] > 0)) continue
            if (res.hp[s] <= 0) alive[s] = 0
            else left[s] = res.ehp[s]
        }
        first += rounds
    }
    let won = 0
    for (let s = 0; s < samples; s++) if (alive[s] && left[s] <= 0) won++
    return won / samples
}

/** floor 之后的 count 个首领层 */
export function nextBossFloors(floor: number, count: number): number[] {
    const out: number[] = []
    for (let g = floor + 1; out.length < count; g++) if (isBossFloor(g)) out.push(g)
    return out
}

/** 非首领层的遭遇种类（宝箱含遗物盒） */
export const MIX_KINDS = ['normal', 'elite', 'choice', 'loot'] as const
export type MixKind = (typeof MIX_KINDS)[number]
/** 服务器 event_weights：F1 行与深层行（普通、精英、奇遇、宝箱+遗物），按 12 层半衰期混合 */
const FIRST_MIX = [36, 3, 12, 49]
const DEEP_MIX = [41, 12, 13, 34]

export function floorMix(floor: number): Record<MixKind, number> {
    const u = 1 - 2 ** (-Math.max(0, floor - 1) / 12)
    const w = FIRST_MIX.map((first, i) => first + (DEEP_MIX[i] - first) * u)
    const total = w.reduce((a, b) => a + b, 0)
    return { normal: w[0] / total, elite: w[1] / total, choice: w[2] / total, loot: w[3] / total }
}

const ODDS_PART = /(箱|怪|精|奇|遗)(\d+(?:\.\d+)?)%/g
const ODDS_KIND: Record<string, MixKind> = { 箱: 'loot', 遗: 'loot', 怪: 'normal', 精: 'elite', 奇: 'choice' }

/** 下一层预览（run.next_floor，如「📊 F247：箱34% 怪41% 精12% 奇13%｜…」）中的遭遇概率；没有概率时为 null */
export function parseFloorOdds(text: string | null | undefined): Record<MixKind, number> | null {
    const head = (text || '').split('｜')[0]
    const parts = [...head.matchAll(ODDS_PART)]
    if (!parts.length) return null
    const odds: Record<MixKind, number> = { normal: 0, elite: 0, choice: 0, loot: 0 }
    for (const [, label, value] of parts) odds[ODDS_KIND[label]] += Number(value) / 100
    return odds
}
