// ============================================================================
// combat.ts — 实战战斗估算（移植自 combat.py）
//
// 由局内状态构造 Build，用 model.ts 的引擎对当前战斗做蒙特卡洛。两个校准系数
// （实测 / 公式伤害，从战斗日志在线学习）吸收公式未暴露的部分；日志伤害先除以本局
// 双向伤害倍率（策划的平衡性调整）再与公式比较。
// ============================================================================

import { BEAT_EVERY, BEAT_TAKEN, BOSS_WINDOW, effAtk, effDef, fight, fightSegment, floorMix, foeAt } from './model.ts'
import type { Build, Foe } from './model.ts'
import { buildEffects } from './heirlooms.ts'
import { defaultRng } from './rng.ts'
import type { Rng } from './rng.ts'
import type { Json } from './types.ts'

const BERSERK_CAP = 0.75

export function rawHit(atk: number, defense: number): number {
    return (atk * 100.0) / (100.0 + 2.5 * Math.max(0.0, defense))
}

const SEG_ROUND = /^R(\d+)：/
const SEG_PLAYER = /^你打 (\d+)/
const SEG_ENEMY = /^怪打 (\d+)（你剩\d+，盾挡(\d+)）/

export interface BattleHits {
    player: [number, number][]
    enemy: [number, number][]
}

/** 提取非暴击的玩家命中与普通敌人命中（含护盾吸收部分） */
export function parseBattleLog(log: string[]): BattleHits {
    const out: BattleHits = { player: [], enemy: [] }
    for (const line of log) {
        const m = SEG_ROUND.exec(line)
        const rnd = m ? Number(m[1]) : 0
        for (const seg of line.replace(SEG_ROUND, '').split(' ｜ ')) {
            const pm = SEG_PLAYER.exec(seg)
            if (pm) {
                out.player.push([rnd, Number(pm[1])])
                continue
            }
            const em = SEG_ENEMY.exec(seg)
            if (em) out.enemy.push([rnd, Number(em[1]) + Number(em[2])])
        }
    }
    return out
}

export interface CalibrationData {
    k_player: number
    k_enemy: number
    n_player: number
    n_enemy: number
}

/** 实测 / 预测伤害的滑动估计，跨会话保存 */
export class Calibration {
    kPlayer: number
    kEnemy: number
    nPlayer: number
    nEnemy: number
    alpha = 0.05

    constructor(data: Partial<CalibrationData> = {}) {
        this.kPlayer = data.k_player ?? 1.0
        this.kEnemy = data.k_enemy ?? 1.0
        this.nPlayer = data.n_player ?? 0
        this.nEnemy = data.n_enemy ?? 0
    }

    toJSON(): CalibrationData {
        return { k_player: this.kPlayer, k_enemy: this.kEnemy, n_player: this.nPlayer, n_enemy: this.nEnemy }
    }

    /**
     * 决策用的系数。公式已核实精确；学到的偏差多半来自临时增益（减伤、护盾），
     * 因此不允许它让战斗显得比公式更轻松。
     */
    conservative(): [number, number] {
        return [Math.min(1.0, this.kPlayer), Math.max(1.0, this.kEnemy)]
    }

    private updatePlayer(ratio: number): void {
        this.nPlayer += 1
        const w = Math.max(this.alpha, 1.0 / this.nPlayer)
        this.kPlayer = (1 - w) * this.kPlayer + w * ratio
    }

    private updateEnemy(ratio: number): void {
        this.nEnemy += 1
        const w = Math.max(this.alpha, 1.0 / this.nEnemy)
        this.kEnemy = (1 - w) * this.kEnemy + w * ratio
    }

    /** 用一场战斗日志更新系数；build 为预测这场战斗时使用的 Build */
    observe(build: Build, foe: Json, log: string[]): void {
        const hits = parseBattleLog(log)
        const bossMult = foe.boss ? 1 + Math.min(1.2, build.bossBonus) : 1.0
        const pPred =
            rawHit(effAtk(build), foe.def * (1 - Math.min(build.pierceCap, build.pierce))) * build.dealtMult * bossMult
        const ePred = rawHit(foe.atk, effDef(build)) * build.takenMult * (1 - build.reduction)
        for (const [rnd, dmg] of hits.player) {
            // 第 1 回合带一次性加成，不参与校准
            if (rnd > 1 && pPred > 0) {
                const berserk = 1 + Math.min(BERSERK_CAP, (rnd - 1) * build.berserk)
                this.updatePlayer(dmg / (pPred * berserk))
            }
        }
        for (const [rnd, dmg] of hits.enemy) {
            let heavy = foe.boss && rnd % BOSS_WINDOW === 0 ? 1.26 : 1.0
            if (build.beat && rnd % BEAT_EVERY === 0) heavy *= BEAT_TAKEN
            if (ePred > 0) this.updateEnemy(dmg / (ePred * heavy))
        }
    }
}

export interface FightEstimate {
    /** 视界内敌人死亡的概率 */
    win: number
    /** 视界内玩家死亡的概率 */
    death: number
    /** 存活样本结束时的平均 HP + 护盾 */
    hpAfter: number
}

/** 当前时刻的战斗 Build：局内实时属性 + 永久词条。二级属性为服务器汇总值（装备、祝福、局内道具）；
 * 打击减伤 mitigation 自 2026-11-25 起提供 */
/** 虹彩永久技能的逃跑加成（风隙羽标：常规上限后再加 20 个百分点）；其余加成已在属性中 */
const RAINBOW_FLEE: Record<string, number> = { prism_wind: 0.2 }

export function rainbowFleeBonus(state: Json): number {
    return ((state.atlas?.rainbow || []) as Json[])
        .filter(r => r.unlocked)
        .reduce((sum, r) => sum + (RAINBOW_FLEE[r.id] ?? 0), 0)
}

/** 伤害公式看到的敌人：防御按本层祝福深层破甲（secondary.condition.depth_shred）削减 */
export function battleEnemy(state: Json, foe: Foe): Foe {
    const shred = Number(state.player?.secondary?.condition?.depth_shred || 0)
    return shred > 0 ? { ...foe, def: foe.def * (1 - shred) } : foe
}

export function runBuild(state: Json, affixBuild: Build): Build {
    const run = state.run
    const sec = state.player?.secondary || {}
    const es = run.effective_stats
    const mult = Number(run.damage_multiplier || 1)
    return {
        ...affixBuild,
        ...buildEffects(state.heirlooms),
        depthShredCap: Number(sec.condition?.depth_cap || 0),
        fleeBonus: rainbowFleeBonus(state),
        hp: run.max_hp,
        atk: es.atk,
        defense: es.def,
        agi: es.agi,
        luk: es.luk,
        pierce: sec.pierce || affixBuild.pierce,
        leech: sec.leech || affixBuild.leech,
        combo: sec.combo || affixBuild.combo,
        bossBonus: sec.boss_bonus || affixBuild.bossBonus,
        critMult: sec.crit_multiplier || affixBuild.critMult,
        reduction:
            sec.mitigation === undefined || sec.mitigation === null ? affixBuild.reduction : Number(sec.mitigation),
        dealtMult: mult,
        takenMult: mult,
        atkPct: 0,
        defPct: 0,
        // 不屈每局一次且状态不可见：按已用掉处理
        lastStand: false
    }
}

function tuned(build: Build, cal: Calibration): Build {
    const [kp, ke] = cal.conservative()
    return { ...build, dealtMult: build.dealtMult * kp, takenMult: build.takenMult * ke }
}

export function estimate(
    build: Build,
    state: Json,
    cal: Calibration,
    samples = 600,
    rng: Rng = defaultRng
): FightEstimate {
    const run = state.run
    const foe = battleEnemy(state, run.event.enemy as Foe)
    const boss = Boolean(foe.boss)
    const [start, rounds] = fightSegment(foe)
    const n = samples
    const res = fight(
        tuned(build, cal),
        new Float64Array(n).fill(run.hp),
        new Float64Array(n).fill(run.shield || 0),
        foe,
        {
            boss,
            ehp: new Float64Array(n).fill(foe.hp),
            startRound: start,
            rounds,
            rng
        }
    )
    let dead = 0
    let won = 0
    let hpSum = 0
    for (let i = 0; i < n; i++) {
        if (res.hp[i] <= 0) dead++
        else {
            if (res.ehp[i] <= 0) won++
            hpSum += res.hp[i] + res.shield[i]
        }
    }
    const alive = n - dead
    return { win: won / n, death: dead / n, hpAfter: alive > 0 ? hpSum / alive : 0 }
}

/**
 * 接下来 fights 层每层都是普通怪、连续作战且中途不回血（连战的情形）时的死亡概率。
 * 用于在连战与单步之间选择。elites：开启「自动打精英」的连战——每场按该层精英占战斗的比例是精英，
 * 打到整备为止；精英在整备时仍存活则该样本的连战停下。
 */
export function chainRisk(
    build: Build,
    state: Json,
    cal: Calibration,
    fights = 4,
    samples = 600,
    mapId = 'mall',
    rng: Rng = defaultRng,
    elites = false
): number {
    const run = state.run
    const b = tuned(build, cal)
    const hp = new Float64Array(samples).fill(run.hp)
    const shield = new Float64Array(samples).fill(run.shield || 0)
    const going = new Uint8Array(samples).fill(1) // 仍在连战（只有精英整备会让连战停下）
    for (let k = 1; k <= fights; k++) {
        const floor = run.floor + k
        const normal = fight(b, hp, shield, foeAt('normal', floor, mapId, b), { boss: false, rng })
        let elite: ReturnType<typeof fight> | null = null
        let met: Uint8Array | null = null
        if (elites) {
            const mix = floorMix(floor)
            const foe: Foe = { ...foeAt('elite', floor, mapId, b), elite: true, rounds_fought: 0 }
            const [first, rounds] = fightSegment(foe)
            elite = fight(b, hp, shield, foe, { boss: false, startRound: first, rounds, rng })
            const share = mix.elite / (mix.elite + mix.normal)
            met = new Uint8Array(samples)
            for (let i = 0; i < samples; i++) met[i] = going[i] && rng.random() < share ? 1 : 0
        }
        for (let i = 0; i < samples; i++) {
            if (!going[i]) continue
            const res = met?.[i] ? elite! : normal
            hp[i] = res.hp[i] <= 0 ? -1.0 : res.hp[i] // 阵亡的样本保持阵亡
            shield[i] = res.shield[i]
            if (met?.[i] && elite!.ehp[i] > 0) going[i] = 0
        }
    }
    return hp.filter(v => v <= 0).length / samples
}
