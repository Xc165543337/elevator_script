// ============================================================================
// markov.ts — 由求解好的马尔可夫决策过程驱动的刷首领策略（移植自 markov_policy.py）
//
// 保留阈值策略在战斗之外的全部逻辑（奇遇、临装、安全箱、收藏品、首层肾上腺素），替换消耗整局资源的决策：
// 遭遇与首领窗口里的回复 / 局内道具 / 烟雾弹 / 逃跑 / 战斗，以及层间的回复 / 单步 / 连战。
// 每次决策：
//   1. 按当前回复与烟雾弹库存取价格（松弛预算的对偶界）；
//   2. 在该价格下评估所有合法操作——战斗按当前敌人与实时 Build 实时模拟，后续价值查表；
//   3. 取最优。连战只要比单步少的期望首领不超过 rushTolerance 就优先（单步只是更慢）。
// 与 Python 版不同：表在后台分片求解（见 solver.ts），不阻塞对局。还没有表时用阈值策略；
// Build 变化后的新表求解期间继续用旧表（实时战斗始终按准确的 Build 模拟）。
// ============================================================================

import { battleEnemy } from './combat.ts'
import type { Calibration } from './combat.ts'
import { applyPrep, compatible, fightPreparations, liveBuild, SPECS, wornMods } from './field.ts'
import type { FieldPrep } from './field.ts'
import { pct } from './format.ts'
import {
    BOSS_WINDOWS,
    Closure,
    DEFAULT_MDP,
    FIELD_HEALS,
    SHOP_HEALS,
    bossAt,
    bossSlice,
    elitePauseKernel,
    elitePauseValue,
    fleeChance,
    fleeHit,
    healApply,
    healItem,
    healUnits,
    inRange,
    interp,
    interpValue,
    priceIndex,
    tableAt,
    tableRow,
    withCalibration
} from './mdp.ts'
import type { HealItem, MdpConfig, ValueTables } from './mdp.ts'
import { BOSS_WINDOW, bossWinChance, fight, fightSegment, foeAt, nextBossFloors, pausesAfterSegment } from './model.ts'
import type { Build, Foe } from './model.ts'
import { BossFarmPolicy, KEEP_IN_BAG, decision, paramsKey } from './policy.ts'
import type { Decision, PolicyConfig } from './policy.ts'
import { defaultRng, seededRng } from './rng.ts'
import { blockingSolver } from './solver.ts'
import type { TableSolver } from './solver.ts'
import type { Rng } from './rng.ts'
import type { Json } from './types.ts'

/** 为了我所期待的 */
export const WISH = 'field_wish'

export interface MarkovConfig {
    /** 更浅的楼层很简单，Build 也还在成形 */
    fromFloor: number
    /** 连战可比单步少的期望首领（每次连战都要付，只用来吸收蒙特卡洛噪声） */
    rushTolerance: number
    /** 回复 / 逃跑须多出的期望首领 */
    healMargin: number
    /** 局内增益 / 手雷（无预算价格）须多出的期望首领 */
    prepMargin: number
    /** 最好的几个单项准备两两组合 */
    prepPairs: number
    /** 普通战斗死亡率达到此值才搜索局内准备 */
    prepMinDeath: number
    /** 预计还能击败的首领数低于此值时撤离（打不过的首领墙；0 = 从不） */
    extractBelow: number
    outlookBosses: number
    /** 实时评估时每个生命格点的战斗次数 */
    liveSamples: number
    /** 评估精英整备时每个 (生命, 精英血量) 格点的战斗次数 */
    pauseSamples: number
    wishGain: number
    wishMinDeath: number
    wishMaxLoot: number
    /** 属性相对变化超过此值才重新求解（实时战斗始终用准确的 Build） */
    buildTolerance: number
    mdp: MdpConfig
}

export const DEFAULT_MARKOV: MarkovConfig = {
    fromFloor: 30,
    rushTolerance: 0.002,
    healMargin: 0.002,
    prepMargin: 0.01,
    prepPairs: 3,
    prepMinDeath: 0.001,
    extractBelow: 0.3,
    outlookBosses: 3,
    liveSamples: 300,
    pauseSamples: 48,
    wishGain: 0.15,
    wishMinDeath: 0.05,
    wishMaxLoot: 15_000_000,
    buildTolerance: 0.1,
    mdp: DEFAULT_MDP
}

interface HealOption {
    item: HealItem
    decision: Decision
}

const HEAL_TEXT = /回复 (\d+)% 最大生命/

/** 即时局内道具（回复和 / 或护盾）对应的 HealItem；其他道具为 null */
export function fieldHealSpec(item: Json): HealItem | null {
    const raw = SPECS[item.id ?? '']
    if (raw) {
        if (raw.mode !== 'instant' || raw.cost_hp || !(raw.heal || raw.shield)) return null
        return healItem(raw.name, { heal: raw.heal ?? 0, shieldAdd: raw.shield ?? 0 })
    }
    const spec = FIELD_HEALS[item.name ?? '']
    if (spec) return spec
    if (item.healing) {
        const m = HEAL_TEXT.exec(item.description || '')
        return m ? healItem(item.name || '?', { heal: Number(m[1]) / 100 }) : null
    }
    return null
}

/** 回复预算（最大生命单位）：携带的药品 + 即时局内回复 */
export function healStock(run: Json): number {
    const maxHp = Number(run.max_hp)
    let total = 0
    for (const it of (run.bag || []) as Json[]) {
        const spec = SHOP_HEALS[it.id]
        if (spec) total += healUnits(spec, maxHp) * (it.quantity || 1)
    }
    for (const it of (run.field || []) as Json[]) {
        const spec = fieldHealSpec(it)
        if (spec) total += healUnits(spec, maxHp) * (it.quantity || 0)
    }
    return total
}

/**
 * 下一层进入时 HP 会上升（缓释再生液、局内再生）。服务器的连战停止规则（HP ≤ 50%）读取的是回复之后的 HP，
 * 表里没有建模——低血时发起的连战可能在刚过半血时自动打一只深层怪。
 */
export function entryHealPending(run: Json): boolean {
    if (run.recovery?.left) return true
    const regen = [...(run.field_buffs || []), ...(run.field_slots || [])].some(
        (e: Json) => (SPECS[e?.id ?? '']?.mods?.regen ?? 0) > 0
    )
    return regen
}

export function smokeCount(run: Json): number {
    return ((run.bag || []) as Json[]).filter(it => it.id === 'smoke').reduce((s, it) => s + (it.quantity || 1), 0)
}

export function effectiveHp(run: Json): number {
    return (run.hp + (run.shield || 0)) / run.max_hp
}

/** 最小二乘单调不减拟合（相邻违例合并）：状态价值只会随 HP 增长，去掉相邻格点间的蒙特卡洛噪声 */
export function isotonic(y: ArrayLike<number>): Float64Array {
    const blocks: [number, number][] = []
    for (let i = 0; i < y.length; i++) {
        blocks.push([y[i], 1])
        while (blocks.length > 1 && blocks[blocks.length - 2][0] > blocks[blocks.length - 1][0]) {
            const [v2, n2] = blocks.pop()!
            const [v1, n1] = blocks.pop()!
            blocks.push([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
        }
    }
    const out = new Float64Array(y.length)
    let k = 0
    for (const [v, n] of blocks) for (let i = 0; i < n; i++) out[k++] = v
    return out
}

/** 求解与缓存用的 Build：属性取整，伤害倍率按保守方向保留两位小数 */
export function quantize(b: Build): Build {
    return {
        ...b,
        hp: Math.round(b.hp),
        atk: Math.round(b.atk),
        defense: Math.round(b.defense),
        agi: Math.round(b.agi),
        luk: Math.round(b.luk),
        dealtMult: Math.floor(b.dealtMult * 100) / 100,
        takenMult: Math.ceil(b.takenMult * 100) / 100
    }
}

/** 两个 Build 在相对容差内相同（如在线校准的漂移） */
export function closeBuilds(a: Build, b: Build, tol: number): boolean {
    for (const key of Object.keys(a) as (keyof Build)[]) {
        const x = a[key]
        const y = b[key]
        if (typeof x === 'number' && typeof y === 'number') {
            if (Math.abs(x - y) > tol * Math.max(Math.abs(x), Math.abs(y), 1)) return false
        } else if (key !== 'label' && x !== y) return false
    }
    return true
}

/**
 * 价值表的后台求解与持有。同一时间只保留一套表（F4000 约 300 MB）。
 * 求解约 5 分钟，所以运行器在页面内跨多次「开始」共用同一个 Planner。
 */
export class Planner {
    private current: ValueTables | null = null
    private pendingBuild: Build | null = null
    private pendingMap = ''
    pending: Promise<void> | null = null
    /** 求解失败（如内存不足）后不再重试，策略保持阈值规则 */
    failure: string | null = null

    constructor(
        private readonly cfg: MarkovConfig,
        private readonly solver: TableSolver,
        /** 求解进度与结果的去处（换成当前运行器的日志） */
        public log: (msg: string) => void = () => {}
    ) {}

    /** 确保当前 Build 的表已有或正在求解；返回求解完成的 Promise（不需要求解时立即完成） */
    request(build: Build, mapId: string): Promise<void> {
        if (this.failure) return Promise.resolve()
        const tol = this.cfg.buildTolerance
        const cur = this.current
        if (cur && cur.mapId === mapId && closeBuilds(cur.build, build, tol)) return Promise.resolve()
        if (this.pending && this.pendingMap === mapId && closeBuilds(this.pendingBuild!, build, tol))
            return this.pending
        if (this.pending) return this.pending // 一次只解一套；完成后的下一次请求会按新 Build 再解
        const target = quantize(build)
        this.pendingBuild = target
        this.pendingMap = mapId
        this.log(
            `马尔可夫：后台求解价值表（攻 ${target.atk}，防 ${target.defense}，敏 ${target.agi}，血 ${target.hp}）`
        )
        this.pending = this.solver
            .solve(target, mapId, this.cfg.mdp, msg => this.log(msg))
            .then(tables => {
                this.current = tables
                this.log(`马尔可夫：价值表就绪（${tables.solveSeconds.toFixed(0)} 秒）`)
            })
            .catch((err: unknown) => {
                this.failure = err instanceof Error ? err.message : String(err)
                this.log(`马尔可夫：求解失败，本次运行改用阈值规则：${this.failure}`)
            })
            .finally(() => {
                this.pending = null
                this.pendingBuild = null
            })
        return this.pending
    }

    /** 当前地图可用的表（可能属于稍早的 Build） */
    get(mapId: string): ValueTables | null {
        return this.current && this.current.mapId === mapId ? this.current : null
    }

    /** 测试用：直接放入一套表 */
    set(tables: ValueTables): void {
        this.current = tables
    }
}

export class MarkovPolicy extends BossFarmPolicy {
    readonly mcfg: MarkovConfig
    readonly planner: Planner
    rng: Rng = defaultRng
    /** 已请求「为了我所期待的」预览；下一次决策确认 */
    private wishSent = false
    private bossOdds = new Map<string, number>()

    constructor(
        cal: Calibration,
        config: Partial<PolicyConfig> = {},
        markov: Partial<MarkovConfig> = {},
        planner?: Planner
    ) {
        super(cal, config)
        this.mcfg = { ...DEFAULT_MARKOV, ...markov }
        this.planner = planner ?? new Planner(this.mcfg, blockingSolver)
    }

    // ------------------------------------------------------------------ 表
    private calibrated(build: Build): Build {
        const [kp, ke] = this.cal.conservative()
        return withCalibration(build, kp, ke)
    }

    /** 不含短期加成（不显示为增益）的 Build：敏捷封顶为面板 + 佩戴的临装（冰红茶只加两层） */
    tableBuild(state: Json): Build {
        const build = this.lastingBuild(state)
        const run = state.run
        const base = run.stats?.agi
        if (base === undefined || base === null) return build
        const lasting = Math.max(0, base + (wornMods(run).agi ?? 0))
        return { ...build, agi: Math.min(build.agi, lasting) }
    }

    /** 为当前 Build 准备价值表（后台求解）。开局用完肾上腺素后就开始，到 fromFloor 时通常已就绪 */
    prepare(state: Json): Promise<void> {
        const run = state.run
        if (!run || run.floor < 5) return Promise.resolve()
        return this.planner.request(this.calibrated(this.tableBuild(state)), run.map || 'mall')
    }

    private tables(state: Json): ValueTables | null {
        const run = state.run
        if (run.floor < this.mcfg.fromFloor) return null
        const t = this.planner.get(run.map || 'mall')
        return t && inRange(t, run.floor) ? t : null
    }

    // ------------------------------------------------------------------ 辅助
    private healOptions(run: Json, inCombat: boolean): HealOption[] {
        const floor = run.floor
        const out: HealOption[] = []
        const seen = new Set<string>()
        for (const it of (run.bag || []) as Json[]) {
            const spec = SHOP_HEALS[it.id]
            if (!spec || seen.has(spec.id) || (spec.combatOnly && !inCombat)) continue
            const params = { index: it.index }
            if (this.itemBlocked(run, spec.id) || !this.allowed(floor, 'use', params)) continue
            seen.add(spec.id)
            out.push({ item: spec, decision: decision('use', params, `使用 ${it.name || spec.id}`) })
        }
        for (const it of (run.field || []) as Json[]) {
            const spec = fieldHealSpec(it)
            const params = { id: it.id }
            if (!spec || !(it.quantity > 0) || seen.has(spec.id) || !this.allowed(floor, 'field_use', params)) continue
            seen.add(spec.id)
            out.push({ item: spec, decision: decision('field_use', params, `局内回复 ${it.name}`) })
        }
        return out
    }

    /** 价格 lam 下最好的回复与「不回复继续」的比较：[选项或 null, Q(最好的回复), Q(不回复)] */
    private choose(
        e: number,
        lam: number,
        maxHp: number,
        curve: Float64Array,
        grid: Float64Array,
        options: HealOption[]
    ): [HealOption | null, number, number] {
        const closure = new Closure(
            options.map(o => o.item),
            grid,
            maxHp,
            Float64Array.of(lam),
            this.mcfg.mdp.healIterations
        )
        const healed = closure.apply(curve, 1, grid.length)
        const qNone = interpValue(curve, 0, grid, e)
        let best: HealOption | null = null
        let qBest = -Infinity
        for (const o of options) {
            const target = healApply(o.item, e, maxHp)
            if (target <= e + 1e-9) continue
            const q = interpValue(healed, 0, grid, target) - lam * healUnits(o.item, maxHp)
            if (q > qBest) {
                best = o
                qBest = q
            }
        }
        return [best, qBest, qNone]
    }

    private liveBuildOf(state: Json): Build {
        const folded = state.player?.secondary?.mitigation !== undefined && state.player?.secondary?.mitigation !== null
        return this.calibrated(liveBuild(state.run, this.currentBuild(state), folded))
    }

    /**
     * 现在战斗的 [价值, 死亡概率]，对每个有效生命格点。普通怪：打到回合上限，后续查 ready 表；精英只打到整备，
     * 整备时用回复后打完剩余回合估值；首领：一个窗口，后续查首领表。死亡概率只覆盖这一段。
     */
    private fightCurve(
        state: Json,
        t: ValueTables,
        li: number,
        build: Build,
        enemy: Foe
    ): [Float64Array, Float64Array] {
        const g = state.run.floor
        const grid = t.grid
        const N = grid.length
        const S = this.mcfg.liveSamples
        const count = (N - 1) * S
        const start = new Int32Array(count)
        const hp = new Float64Array(count)
        const shield = new Float64Array(count)
        for (let k = 0; k < count; k++) {
            start[k] = 1 + Math.floor(k / S)
            hp[k] = Math.min(grid[start[k]], 1) * build.hp
            shield[k] = Math.max(grid[start[k]] - 1, 0) * build.hp
        }
        const ehp = new Float64Array(count).fill(enemy.hp)
        const readyRow = tableRow(t, t.ready, g, li)
        const after = new Float64Array(count)
        let res
        if (enemy.boss) {
            const fought = Number(enemy.rounds_fought || 0)
            const window = Math.floor(fought / BOSS_WINDOW) + 1
            const bossMax = Number(enemy.max_hp || enemy.hp)
            res = fight(build, hp, shield, enemy, {
                boss: true,
                ehp,
                startRound: fought + 1,
                rounds: BOSS_WINDOW - (fought % BOSS_WINDOW),
                rng: this.rng
            })
            const next = bossSlice(t, g, Math.min(window + 1, BOSS_WINDOWS), li)
            const winValue = 1 + interpValue(readyRow, 0, grid, 1.0)
            for (let k = 0; k < count; k++) {
                after[k] =
                    res.ehp[k] <= 0
                        ? winValue
                        : bilinear(next, grid, t.bossGrid, (res.hp[k] + res.shield[k]) / build.hp, res.ehp[k] / bossMax)
            }
        } else {
            const [first, rounds] = fightSegment(enemy)
            res = fight(build, hp, shield, enemy, { boss: false, ehp, startRound: first, rounds, rng: this.rng })
            const pause = pausesAfterSegment(enemy) ? this.elitePause(state, t, li, build, enemy) : null
            const eliteMax = Number(enemy.max_hp || enemy.hp)
            for (let k = 0; k < count; k++) {
                const e1 = (res.hp[k] + res.shield[k]) / build.hp
                after[k] =
                    pause && res.ehp[k] > 0
                        ? bilinear(pause, grid, t.bossGrid, e1, Math.max(res.ehp[k], 0) / eliteMax)
                        : interpValue(readyRow, 0, grid, e1)
            }
        }
        const sums = new Float64Array(N)
        const alive = new Float64Array(N)
        for (let k = 0; k < count; k++) {
            if (res.hp[k] > 0) {
                sums[start[k]] += after[k]
                alive[start[k]] += 1
            }
        }
        const value = new Float64Array(N)
        const death = new Float64Array(N)
        death[0] = 1
        for (let i = 1; i < N; i++) {
            value[i] = sums[i] / S
            death[i] = 1 - alive[i] / S
        }
        return [isotonic(value), death]
    }

    /**
     * 预计还能击败的首领数（P1 + P1·P2 + …，接下来 outlookBosses 个首领）。每个胜率都假设每个窗口前回满、
     * 用不含短期增益的 Build：是上界，所以由它触发的撤离只在完美回复也过不去时才发生。
     */
    protected bossOutlook(state: Json, floor: number): number {
        const build = quantize(this.calibrated(this.lastingBuild(state)))
        const mapId = state.run.map || 'mall'
        let total = 0
        let chain = 1
        for (const bossFloor of nextBossFloors(floor, this.mcfg.outlookBosses)) {
            const key = `${JSON.stringify(build)}|${mapId}|${bossFloor}`
            let p = this.bossOdds.get(key)
            if (p === undefined) {
                p = bossWinChance(build, foeAt('boss', bossFloor, mapId, build), { rng: seededRng(0) })
                this.bossOdds.set(key, p)
            }
            chain *= p
            total += chain
        }
        return total
    }

    /** 打不过的首领：撤离（首领战中可撤离），否则用烟雾弹（同样算撤离）带回背包，而不是硬打 */
    private hopelessBoss(
        state: Json,
        ctr: Map<string, Json>,
        build: Build,
        enemy: Foe,
        summary: string
    ): Decision | null {
        const run = state.run
        if (this.mcfg.extractBelow <= 0 || this.greedy(run)) return null
        const win = bossWinChance(
            build,
            { ...enemy, hp: Number(enemy.max_hp || enemy.hp) },
            { ehp: enemy.hp, startRound: Number(enemy.rounds_fought || 0) + 1, rng: seededRng(0) }
        )
        const outlook = win * (1 + this.bossOutlook(state, run.floor))
        if (outlook >= this.mcfg.extractBelow) return null
        const why = `${summary}：每窗口回满的胜率 ${pct(win)}，预计还能击败 ${outlook.toFixed(2)} 个首领`
        if (ctr.has('extract')) return decision('extract', {}, `${why} → 撤离带回背包`)
        const smoke = MarkovPolicy.bagItem(run, 'smoke')
        if (smoke && this.allowed(run.floor, 'use', { index: smoke.index })) {
            return decision('use', { index: smoke.index }, `${why} → 烟雾弹脱出（算作撤离）`)
        }
        return null
    }

    /**
     * 即将到来的精英整备在价格 li 下的值 [生命][精英血量]：用实时库存回复后打完剩余回合。
     * 防护 / 蓄力次数按届时已用完处理（保守）。
     */
    private elitePause(state: Json, t: ValueTables, li: number, build: Build, enemy: Foe): Float64Array {
        const run = state.run
        const heal = new Closure(
            this.healOptions(run, true).map(o => o.item),
            t.grid,
            Number(run.max_hp),
            Float64Array.of(t.lambdas[li]),
            this.mcfg.mdp.healIterations
        )
        const foe = { ...enemy, hp: Number(enemy.max_hp || enemy.hp), rounds_fought: 0 }
        const later = { ...build, guardHits: 0, burstHits: 0 }
        const lands = elitePauseKernel(later, foe, t.grid, t.bossGrid, this.mcfg.pauseSamples, this.rng)
        return elitePauseValue(lands, tableRow(t, t.ready, run.floor, li), 1, heal, t.bossGrid.length)
    }

    // ------------------------------------------------------------------ 遭遇
    protected fightDecision(state: Json, ctr: Map<string, Json>): Decision {
        const enemy = state.run.event.enemy
        const t = enemy.ambush ? null : this.tables(state)
        return t ? this.encounter(state, ctr, t) : super.fightDecision(state, ctr)
    }

    /**
     * 回复 / 局内增益 / 手雷 / 烟雾弹 / 逃跑 / 战斗中期望首领最多者。回复与烟雾弹按整局预算定价；局内增益不定价
     * 但须多出 prepMargin。每个选项的价值都已包含之后可能的最佳回复。
     */
    private encounter(state: Json, ctr: Map<string, Json>, t: ValueTables): Decision {
        const run = state.run
        const enemy = battleEnemy(state, run.event.enemy as Foe)
        const boss = Boolean(enemy.boss)
        const g = run.floor
        const e = effectiveHp(run)
        const maxHp = Number(run.max_hp)
        const grid = t.grid
        const kind = boss ? 'boss' : enemy.elite ? 'elite' : 'normal'
        const stock = healStock(run)
        const values = boss
            ? bossAt(
                  t,
                  g,
                  Math.floor(Number(enemy.rounds_fought || 0) / BOSS_WINDOW) + 1,
                  e,
                  enemy.hp / Math.max(1, Number(enemy.max_hp || enemy.hp))
              )
            : tableAt(t, kind === 'elite' ? t.encElite : t.encNormal, g, e)
        let smoke = boss ? null : MarkovPolicy.bagItem(run, 'smoke') // 对首领用烟雾弹会结束本局
        if (smoke && !this.allowed(g, 'use', { index: smoke.index })) smoke = null
        const smokes = smokeCount(run)
        const li = priceIndex(t, values, stock, smokes)
        const lam = t.lambdas[li]
        const mu = t.smokePrices[li]
        const heals = this.healOptions(run, true)
        const closure = new Closure(
            heals.map(o => o.item),
            grid,
            maxHp,
            Float64Array.of(lam),
            this.mcfg.mdp.healIterations
        )
        const withHeals = (curve: Float64Array) => interpValue(closure.apply(curve, 1, grid.length), 0, grid, e)

        const build = this.liveBuildOf(state)
        const [fightCurve, deathCurve] = this.fightCurve(state, t, li, build, enemy)
        const death = interpValue(deathCurve, 0, grid, e)
        const tag = `${kind} ${enemy.name}`
        const summary = `${tag}：死亡 ${pct(death)}，回复价 ${lam}，库存 ${stock.toFixed(1)}，烟雾弹 ${smokes} @ ${mu}`
        if (boss) {
            const bank = this.hopelessBoss(state, ctr, build, enemy, summary)
            if (bank) return bank
        }

        // 不消耗定价回复的选项：直接战斗、逃跑、局内增益、手雷
        let curve = fightCurve
        let fleeQ = -Infinity
        const readyRow = tableRow(t, t.ready, g, li)
        if (ctr.has('flee') && !boss) {
            const m = /(\d+)%/.exec(ctr.get('flee').label || '')
            const p = m ? Number(m[1]) / 100 : fleeChance(build, g)
            const hit = fleeHit(build, enemy)
            const fleeCurve = new Float64Array(grid.length)
            for (let i = 1; i < grid.length; i++) {
                const failed = interpValue(fightCurve, 0, grid, Math.max(grid[i] - hit, 0))
                fleeCurve[i] = p * readyRow[i] + (1 - p) * failed
            }
            fleeQ = interpValue(fleeCurve, 0, grid, e)
            curve = fightCurve.map((v, i) => Math.max(v, fleeCurve[i]))
        }
        const qBase = withHeals(curve)
        let bestPrep: [string, Record<string, unknown>, string] | null = null
        let qPrep = -Infinity
        if (boss || death >= this.mcfg.prepMinDeath) {
            // 安全的普通战斗不会因增益而更好
            ;[bestPrep, qPrep] = this.bestPreparation(state, t, li, build, enemy, withHeals)
        }
        const [option, qHeal, qGo] = this.choose(e, lam, maxHp, curve, grid, heals)
        const qSmoke = smoke ? interpValue(readyRow, 0, grid, e) - mu : -Infinity
        const healed = Math.max(e, ...heals.map(o => healApply(o.item, e, maxHp)))
        if (this.wishReady(state) && interpValue(deathCurve, 0, grid, healed) >= this.mcfg.wishMinDeath) {
            const qWish = boss ? 1 + interpValue(readyRow, 0, grid, 1.0) : interpValue(readyRow, 0, grid, e)
            const qOther = Math.max(qBase, qHeal, qPrep, qSmoke)
            if (qWish - qOther >= this.mcfg.wishGain * qWish) {
                this.wishSent = true
                return decision(
                    'field_use',
                    { id: WISH },
                    `${summary} → 为了我所期待的（${qWish.toFixed(2)} 对 ${qOther.toFixed(2)}）`
                )
            }
        }
        const qBest = Math.max(qBase, qHeal, qPrep)
        if (smoke && qSmoke > qBest + this.mcfg.healMargin) {
            return decision(
                'use',
                { index: smoke.index },
                `${summary} → 烟雾弹（${qSmoke.toFixed(2)} 对 ${qBest.toFixed(2)}）`
            )
        }
        if (bestPrep && qPrep > Math.max(qBase, qHeal) + this.mcfg.prepMargin) {
            return decision(
                bestPrep[0],
                bestPrep[1],
                `${summary} → ${bestPrep[2]}（${qPrep.toFixed(2)} 对 ${qBase.toFixed(2)}）`
            )
        }
        if (option && qHeal > qGo + this.mcfg.healMargin) {
            return decision(
                option.decision.command,
                option.decision.params,
                `${summary} → ${option.decision.reason}（${qHeal.toFixed(2)} 对 ${qGo.toFixed(2)}）`
            )
        }
        const fightQ = interpValue(fightCurve, 0, grid, e)
        if (fleeQ > fightQ + this.mcfg.healMargin) {
            return decision('flee', {}, `${summary} → 逃跑（${fleeQ.toFixed(2)} 对 ${fightQ.toFixed(2)}）`)
        }
        return decision('fight', {}, `${summary} → 战斗（${fightQ.toFixed(2)}）`)
    }

    /**
     * 局内道具里有「为了我所期待的」，且清空的东西都不重要：没有仍缺的收藏品、没有正在带回的遗物盒、
     * 背包与安全箱的物资不到 wishMaxLoot
     */
    private wishReady(state: Json): boolean {
        const run = state.run
        if (!((run.field || []) as Json[]).some(it => it.id === WISH && (it.quantity || 0) > 0)) return false
        if (!this.allowed(run.floor, 'field_use', { id: WISH })) return false
        const held = [...(run.bag || []), ...(run.safe || [])] as Json[]
        const missing = BossFarmPolicy.missingCollectibles(state)
        if (held.some(it => missing.has(it.id))) return false
        const loot = held
            .filter(it => !KEEP_IN_BAG.has(it.id))
            .reduce((s, it) => s + Number(it.value || 0) * (it.quantity || 1), 0)
        return loot < this.mcfg.wishMaxLoot && !this.banking
    }

    protected decideRaw(state: Json): Decision {
        const sent = this.wishSent
        this.wishSent = false
        const preview = state.wish_preview || {}
        if (sent && preview.token && (state.run.event || {}).enemy) {
            return decision(
                'wish_confirm',
                { token: preview.token },
                `确认发动「为了我所期待的」：${preview.enemy}（清空本局拾取 ${Number(preview.value || 0).toLocaleString()}）`
            )
        }
        if (sent) this.banned.add(`${state.run.floor}|field_use|${paramsKey({ id: WISH })}`) // 没有拿到预览：本层不再请求
        return super.decideRaw(state)
    }

    /** 局内增益、两两组合与手雷中最好的第一步（含之后的回复）。返回 [(命令, 参数, 说明) 或 null, 值] */
    private bestPreparation(
        state: Json,
        t: ValueTables,
        li: number,
        build: Build,
        enemy: Foe,
        withHeals: (curve: Float64Array) => number
    ): [[string, Record<string, unknown>, string] | null, number] {
        const run = state.run
        const floor = run.floor
        const preps = fightPreparations(run).filter(p => this.allowed(floor, 'field_use', { id: p.id }))
        const scored: [number, FieldPrep, Build, Record<string, number>][] = preps.map(prep => {
            const [b, mods] = applyPrep(run, build, prep)
            return [withHeals(this.fightCurve(state, t, li, b, enemy)[0]), prep, b, mods]
        })
        scored.sort((x, y) => y[0] - x[0])
        let best: [string, Record<string, unknown>, string] | null = null
        let bestQ = -Infinity
        if (scored.length) {
            best = ['field_use', { id: scored[0][1].id }, `局内道具 ${scored[0][1].name}`]
            bestQ = scored[0][0]
        }
        const top = scored.slice(0, this.mcfg.prepPairs)
        for (let i = 0; i < top.length; i++) {
            const [, first, b1, mods1] = top[i]
            for (const [, second] of top.slice(i + 1)) {
                if (!compatible(first, second, run)) continue
                const [b2] = applyPrep(run, b1, second, mods1)
                const q = withHeals(this.fightCurve(state, t, li, b2, enemy)[0])
                if (q > bestQ) {
                    best = ['field_use', { id: first.id }, `局内道具 ${first.name}（+ ${second.name}）`]
                    bestQ = q
                }
            }
        }
        const grenade = MarkovPolicy.bagItem(run, 'grenade')
        const bossCapped = enemy.boss && (Number(enemy.grenades || 0) >= 2 || enemy.hp <= 1)
        if (grenade && !bossCapped && this.allowed(floor, 'use', { index: grenade.index })) {
            const cut = Math.max(1, Math.floor(enemy.hp * (enemy.boss ? 0.12 : 0.3)))
            const weakened = { ...enemy, hp: Math.max(enemy.boss ? 1 : 0, enemy.hp - cut) }
            const q = withHeals(this.fightCurve(state, t, li, build, weakened)[0])
            if (q > bestQ) {
                best = ['use', { index: grenade.index }, '手雷']
                bestQ = q
            }
        }
        return [best, bestQ]
    }

    // ------------------------------------------------------------------ 层间
    protected advance(state: Json, ctr: Map<string, Json>): Decision {
        const run = state.run
        const t = this.tables(state)
        if (!t) return super.advance(state, ctr)
        if (this.cfg.maxFloor && run.floor >= this.cfg.maxFloor && ctr.has('extract')) {
            return decision('extract', {}, `到达最大楼层 ${this.cfg.maxFloor}`)
        }
        const g = run.floor
        const e = effectiveHp(run)
        const maxHp = Number(run.max_hp)
        const stock = healStock(run)
        const li = priceIndex(t, tableAt(t, t.ready, g, e), stock, smokeCount(run))
        const lam = t.lambdas[li]
        const grid = t.grid
        const stepRow = tableRow(t, t.step, g, li)
        const rushRow = tableRow(t, t.rush, g, li)
        const qStep = interpValue(stepRow, 0, grid, e)
        const qRush = interpValue(rushRow, 0, grid, e)
        const summary = `单步 ${qStep.toFixed(2)} 连战 ${qRush.toFixed(2)}，回复价 ${lam}，库存 ${stock.toFixed(1)}`
        if (this.mcfg.extractBelow > 0 && ctr.has('extract') && !this.greedy(run)) {
            const outlook = this.bossOutlook(state, g)
            if (outlook < this.mcfg.extractBelow) {
                return decision('extract', {}, `${summary}：预计还能击败 ${outlook.toFixed(2)} 个首领 → 撤离带回背包`)
            }
        }
        const curve = stepRow.map((v, i) => Math.max(v, rushRow[i]))
        const [option, qHeal, qGo] = this.choose(e, lam, maxHp, curve, grid, this.healOptions(run, false))
        if (option && qHeal > qGo + this.mcfg.healMargin) {
            return decision(
                option.decision.command,
                option.decision.params,
                `${summary} → ${option.decision.reason}（${qHeal.toFixed(2)}）`
            )
        }
        if (ctr.has('rush') && ctr.has('down') && entryHealPending(run)) {
            return decision('down', {}, `${summary} → 单步（进层回复未结算：连战会在回复过半血后自动开打）`)
        }
        if (ctr.has('rush') && (qRush >= qStep - this.mcfg.rushTolerance || !ctr.has('down'))) {
            return decision('rush', {}, `${summary} → 连战`)
        }
        if (ctr.has('down')) return decision('down', {}, `${summary} → 单步`)
        return super.advance(state, ctr)
    }
}

/** 二维表 [生命][敌人血量] 在 (e, b) 处的双线性插值 */
function bilinear(table: Float64Array, grid: Float64Array, enemyGrid: Float64Array, e: number, b: number): number {
    const ie = interp([e], grid)
    const ib = interp([b], enemyGrid)
    const nb = enemyGrid.length
    const e0 = ie.lo[0]
    const b0 = ib.lo[0]
    const we = ie.w[0]
    const wb = ib.w[0]
    return (
        (1 - we) * (1 - wb) * table[e0 * nb + b0] +
        (1 - we) * wb * table[e0 * nb + b0 + 1] +
        we * (1 - wb) * table[(e0 + 1) * nb + b0] +
        we * wb * table[(e0 + 1) * nb + b0 + 1]
    )
}
