// ============================================================================
// mdp.ts — 刷首领的马尔可夫决策过程，逆向归纳求解（移植自 mdp.py）
//
// 目标：一局内正式首领击杀数的期望（每次击杀独立判定 0.5% 虹彩）。
// 状态（每层 g）：有效生命 e =（HP + 护盾）/ 最大生命，取 [0, E_MAX] 上的格点；决策点：
//   ready     —— 无待处理事件：回复（非战斗道具）、下一层或连战；
//   encounter —— 普通怪 / 精英等待处理：回复（任意道具）、战斗、逃跑、烟雾弹；
//   精英整备   —— 精英第 8 回合后：回复后打完剩余回合（求解时临时计算，不存表）；
//   boss      —— 首领第 w 个窗口前（1..4，4 及以后相同），首领剩余血量比例 b。
// 回复与烟雾弹是整局预算，用拉格朗日乘子定价：每个价格对 (λ, μ) 各解一张表，决策时按当前库存取
// 对偶界 V + λH + μS 最小的价格。
//
// 存储布局：每层的表是 [价格][生命格点] 的行主序数组（Float32 存储）；首领表为 [窗口][价格][生命][首领血量]。
// 与 Python 版不同的实现细节：回复的移动矩阵每行只有两个非零元，按稀疏插值计算；蒙特卡洛随机流不同，
// 结果在统计意义上一致。
// ============================================================================

import {
    BOSS_WINDOW,
    ELITE_BREAK,
    ELITE_ROUND_CAP,
    REGULAR_ROUND_CAP,
    effDef,
    fight,
    floorMix,
    foeAt,
    isBossFloor
} from './model.ts'
import type { Build, EnemyKind, Foe } from './model.ts'
import { seededRng } from './rng.ts'
import type { Rng } from './rng.ts'

export const E_MAX = 1.25 // 满血加屏障灌注剂 25% 护盾
export const BOSS_WINDOWS = 4 // 窗口 1-3 不同（首回合加成、越战越勇递增）；4 及以后相同
export const RUSH_FLOORS = 10
export const RUSH_STOP_HP = 0.5 // 服务器在 HP ≤ 50% 时停止连战

/**
 * 金币目标：本局物资（背包 + 安全箱，百万金币）随楼层的期望值，在游戏引擎中用阈值策略实测（每图 3 局，正式局）。
 * 一局背包约千件、每件 2k–48k 金币，某层的总值在各局之间相差不大，所以用一条曲线代替；最后一点之后沿最后一段的斜率延伸
 * （同 Python 版 LOOT_CURVES）
 */
export const LOOT_CURVES: Record<string, [number, number][]> = {
    mall: [
        [0, 0],
        [50, 0.49],
        [100, 0.75],
        [200, 1.71],
        [300, 3.44],
        [500, 6.11],
        [800, 10.14],
        [1000, 12.42],
        [1200, 14.08],
        [1500, 15.99]
    ],
    subway: [
        [0, 0],
        [50, 0.66],
        [100, 1.41],
        [200, 2.93],
        [300, 4.39],
        [500, 7.4],
        [800, 10.12]
    ],
    abyss: [
        [0, 0],
        [50, 1.02],
        [100, 2.12],
        [200, 4.27],
        [300, 6.22],
        [500, 8.47]
    ],
    auction: [
        [0, 0],
        [25, 0.68],
        [50, 1.23],
        [100, 2.39],
        [150, 3.78],
        [200, 4.77],
        [300, 6.22],
        [400, 7.32],
        [500, 8.14],
        [600, 8.78]
    ],
    tide: [
        [0, 0],
        [25, 0.58],
        [50, 1.16],
        [100, 2.14],
        [150, 3.1],
        [200, 3.82],
        [300, 4.94],
        [400, 5.71]
    ],
    zenith: [
        [0, 0],
        [25, 0.6],
        [50, 1.18],
        [100, 2.2],
        [150, 3.11],
        [200, 4.14]
    ]
}
/** 死亡时 4 格安全箱保住的物资（百万金币，实测） */
export const SAFE_KEPT = 0.2

/** 某层的本局期望物资（百万金币）；未实测的地图用商场曲线 */
export function lootAt(mapId: string, floor: number): number {
    const points = LOOT_CURVES[mapId] ?? LOOT_CURVES.mall
    const [lastX, lastY] = points[points.length - 1]
    if (floor <= lastX) {
        for (let i = 1; i < points.length; i++) {
            const [x0, y0] = points[i - 1]
            const [x1, y1] = points[i]
            if (floor <= x1) return y0 + ((y1 - y0) * (floor - x0)) / (x1 - x0)
        }
    }
    const [prevX, prevY] = points[points.length - 2]
    return lastY + Math.max(0, (lastY - prevY) / (lastX - prevX)) * (floor - lastX)
}

/** 每层（0..top+2）的 [进层找到的物资, 在该层死亡损失的物资]，模型单位；非金币目标时全为 0 */
export function lootProfile(config: MdpConfig, mapId: string, top: number): [Float64Array, Float64Array] {
    const gain = new Float64Array(top + 3)
    const risk = new Float64Array(top + 3)
    if (!config.gold) return [gain, risk]
    let prev = 0
    for (let g = 0; g < top + 3; g++) {
        const loot = lootAt(mapId, g)
        gain[g] = loot - prev
        risk[g] = Math.max(loot - SAFE_KEPT, 0)
        prev = loot
    }
    return [gain, risk]
}

/** 按存活部分（kill / draw / cont / lands 的行）拆开的转移核，每个出发状态的死亡概率 */
export function deathOf(rows: number, ...parts: Float64Array[]): Float64Array {
    const out = new Float64Array(rows).fill(1)
    for (const part of parts) {
        const cols = part.length / rows
        for (let r = 0; r < rows; r++) {
            let sum = 0
            for (let c = 0; c < cols; c++) sum += part[r * cols + c]
            out[r] -= sum
        }
    }
    for (let r = 0; r < rows; r++) out[r] = Math.min(1, Math.max(0, out[r]))
    return out
}
const FIELD_DROP_CHANCE = 0.52 // 服务器 FIELD_CHANCE；F1、首领层及其前一层必掉
export const MODEL_VERSION = 8

// ---------------------------------------------------------------------------- 回复道具
/** 提高有效生命的消耗品（数值来自服务器道具规格） */
export interface HealItem {
    id: string
    /** 最大生命的比例 */
    heal: number
    /** 额外的固定回复 */
    flat: number
    /** 护盾至少提升到最大生命的此比例 */
    shieldFloor: number
    /** 额外的固定护盾 */
    shieldAdd: number
    combatOnly: boolean
    /** HP ≤ criticalAt 时改为回复 criticalHeal */
    criticalAt: number
    criticalHeal: number
}

export function healItem(id: string, fields: Partial<Omit<HealItem, 'id'>> = {}): HealItem {
    return {
        id,
        heal: 0,
        flat: 0,
        shieldFloor: 0,
        shieldAdd: 0,
        combatOnly: false,
        criticalAt: 0,
        criticalHeal: 0,
        ...fields
    }
}

/** 名义大小（最大生命单位），即从回复预算中扣除的量 */
export function healUnits(item: HealItem, maxHp: number): number {
    return item.heal + (item.flat + item.shieldAdd) / maxHp + item.shieldFloor
}

/** 使用后的有效生命 */
export function healApply(item: HealItem, e: number, maxHp: number): number {
    let hp = Math.min(e, 1.0)
    let shield = e - hp
    const heal = item.criticalAt && hp <= item.criticalAt ? item.criticalHeal : item.heal
    hp = Math.min(1.0, hp + heal + item.flat / maxHp)
    shield = Math.max(shield, item.shieldFloor) + item.shieldAdd / maxHp
    return Math.min(E_MAX, hp + shield)
}

/** 商店药品（服务器 content.json 的 medicine 与 game.py 中的冰美式 / 冰红茶）。缓释再生液的后续回复不计入。 */
export const SHOP_HEALS: Record<string, HealItem> = Object.fromEntries(
    [
        healItem('coffee', { heal: 0.4 }),
        healItem('ice_tea', { heal: 0.25 }),
        healItem('triage_vial', { heal: 0.55, flat: 12 }),
        healItem('barrier_serum', { heal: 0.5, shieldFloor: 0.25 }),
        healItem('slow_serum', { heal: 0.38 }),
        healItem('critical_serum', { heal: 0.55, criticalAt: 0.35, criticalHeal: 0.9 }),
        healItem('field_surgery', { heal: 1.0, combatOnly: true })
    ].map(h => [h.id, h])
)

/** 即时局内道具（服务器 field_items），按游戏内名称 */
export const FIELD_HEALS: Record<string, HealItem> = Object.fromEntries(
    [
        healItem('速凝绷带', { heal: 0.28 }),
        healItem('试吃能量包', { heal: 0.18, shieldAdd: 16 }),
        healItem('折叠盾片', { shieldAdd: 38 })
    ].map(h => [h.id, h])
)

/** 商场局内道具池：名称 → 调整后的抽取权重（回复 / 吸血 ×1.5，已穿戴的临装 ×0.4）；只需要回复类的名称 */
const MALL_FIELD_POOL: Record<string, number> = { 速凝绷带: 12, 试吃能量包: 12, 折叠盾片: 7, _other: 60 }

/** 进入 floor 时预期捡到的回复量（最大生命单位，商场池） */
export function fieldHealIncome(floor: number, maxHp: number): number {
    const total = Object.values(MALL_FIELD_POOL).reduce((a, b) => a + b, 0)
    let perDrop = 0
    for (const [name, w] of Object.entries(MALL_FIELD_POOL)) {
        if (FIELD_HEALS[name]) perDrop += (w * healUnits(FIELD_HEALS[name], maxHp)) / total
    }
    const guaranteed = floor === 1 || isBossFloor(floor) || isBossFloor(floor + 1)
    return perDrop * (guaranteed ? 1.0 : FIELD_DROP_CHANCE)
}

// ---------------------------------------------------------------------------- 格点与插值
/** 线性插值到格点：x[k] 落在 lo[k] 与 lo[k]+1 之间，权重 (1 - w[k], w[k]) */
export interface Interp {
    lo: Int32Array
    w: Float64Array
}

export function interp(x: ArrayLike<number>, grid: Float64Array): Interp {
    const n = grid.length
    const lo = new Int32Array(x.length)
    const w = new Float64Array(x.length)
    for (let k = 0; k < x.length; k++) {
        const v = Math.min(grid[n - 1], Math.max(grid[0], x[k]))
        let hi = 1
        // 格点数很小（≤ 51）：二分查找第一个 > v 的位置（searchsorted side=right）
        let a = 0
        let b = n
        while (a < b) {
            const mid = (a + b) >> 1
            if (grid[mid] <= v) a = mid + 1
            else b = mid
        }
        hi = Math.min(n - 1, Math.max(1, a))
        lo[k] = hi - 1
        w[k] = (v - grid[hi - 1]) / (grid[hi] - grid[hi - 1])
    }
    return { lo, w }
}

/** 一行数值在 x 处的线性插值 */
export function interpValue(values: ArrayLike<number>, offset: number, grid: Float64Array, x: number): number {
    const { lo, w } = interp([x], grid)
    return (1 - w[0]) * values[offset + lo[0]] + w[0] * values[offset + lo[0] + 1]
}

/** C[m×n] = A[m×k] · B[n×k]ᵀ */
export function mulBT(a: ArrayLike<number>, m: number, k: number, b: ArrayLike<number>, n: number): Float64Array {
    const c = new Float64Array(m * n)
    for (let i = 0; i < m; i++) {
        const ai = i * k
        for (let j = 0; j < n; j++) {
            const bj = j * k
            let s = 0
            for (let t = 0; t < k; t++) s += a[ai + t] * b[bj + t]
            c[i * n + j] = s
        }
    }
    return c
}

/** 稀疏矩阵（压缩行格式） */
export interface Csr {
    rows: number
    indptr: Int32Array
    indices: Int32Array
    data: Float64Array
}

export function toCsr(dense: ArrayLike<number>, rows: number, cols: number): Csr {
    const indptr = new Int32Array(rows + 1)
    const indices: number[] = []
    const data: number[] = []
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const v = dense[r * cols + c]
            if (v !== 0) {
                indices.push(c)
                data.push(v)
            }
        }
        indptr[r + 1] = indices.length
    }
    return { rows, indptr, indices: Int32Array.from(indices), data: Float64Array.from(data) }
}

/** C[m×rows] = A[m×k] · Bᵀ，B 为稀疏矩阵 [rows × k] */
export function mulBTSparse(a: ArrayLike<number>, m: number, k: number, b: Csr): Float64Array {
    const n = b.rows
    const c = new Float64Array(m * n)
    for (let i = 0; i < m; i++) {
        const ai = i * k
        for (let j = 0; j < n; j++) {
            let s = 0
            for (let q = b.indptr[j]; q < b.indptr[j + 1]; q++) s += a[ai + b.indices[q]] * b.data[q]
            c[i * n + j] = s
        }
    }
    return c
}

/** 样本（起点 start[k]，落点 x[k]）按格点线性拆分后的直方图：[nStart][格点] */
function deposit(start: ArrayLike<number>, x: ArrayLike<number>, grid: Float64Array, nStart: number, weight: number) {
    const n = grid.length
    const out = new Float64Array(nStart * n)
    const { lo, w } = interp(x, grid)
    for (let k = 0; k < start.length; k++) {
        out[start[k] * n + lo[k]] += (1 - w[k]) * weight
        out[start[k] * n + lo[k] + 1] += w[k] * weight
    }
    return out
}

// ---------------------------------------------------------------------------- 配置与表
export interface MdpConfig {
    /** 此层以下的值为 0（满套缺刻值夜表可打到 F2000 以上） */
    maxFloor: number
    hpPoints: number
    bossPoints: number
    /** 每个格点的蒙特卡洛战斗次数 */
    samples: number
    /** 普通战斗的转移核由这么多层共用（取最强的怪） */
    kernelStride: number
    /** 首领第 4 窗口（稳定窗口）值迭代的上限 */
    bossIterations: number
    /** 每次决策考虑的回复链长度 */
    healIterations: number
    /** 回复价格（首领 / 最大生命单位）；在 0.03–0.2 之间加密 */
    lambdas: number[]
    /** 烟雾弹价格（首领 / 枚）；最后一个表示没有烟雾弹 */
    smokePrices: number[]
    /** 精英整备转移核每个 (生命, 精英血量) 格点的战斗次数 */
    pauseSamples: number
    /**
     * 同时求解开启「自动打精英」的连战：HP 高于 50% 时连战自己打精英到整备，击杀则继续，否则停在整备
     * （game.py rush）。求解时间约多 40%
     */
    rushElites: boolean
    /**
     * 目标。首领（默认）：每击败一个正式首领 +bossValue。金币：单位为百万金币，一个首领值 bossValue，找到的物资即计入
     * （lootProfile），死亡损失安全箱以外的背包；撤离（层间、战斗中、首领战中）把它带回
     */
    gold: boolean
    bossValue: number
    seed: number
}

export const DEFAULT_MDP: MdpConfig = {
    maxFloor: 4000,
    hpPoints: 51,
    bossPoints: 11,
    samples: 256,
    kernelStride: 4,
    bossIterations: 60,
    healIterations: 8,
    lambdas: [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 10.0],
    smokePrices: [0.0, 0.03, 0.05, 0.07, 0.1, 1000.0],
    pauseSamples: 96,
    rushElites: true,
    gold: false,
    bossValue: 1,
    seed: 0
}

/** 求解结果。价格索引 p 对应 (lambdas[p], smokePrices[p])；Float32 存储 */
export interface ValueTables {
    config: MdpConfig
    build: Build
    mapId: string
    grid: Float64Array
    bossGrid: Float64Array
    lambdas: Float64Array
    smokePrices: Float64Array
    /** [层][价格][生命]：ready 状态的值 */
    ready: Float32Array
    /** 从 ready 状态单步下一层的值 */
    step: Float32Array
    /** 从 ready 状态连战的值 */
    rush: Float32Array
    /** 开启「自动打精英」连战的值（rushElites 关闭时全为 0） */
    rushElite: Float32Array
    encNormal: Float32Array
    encElite: Float32Array
    /** 首领层 → [窗口][价格][生命][首领血量] */
    boss: Map<number, Float32Array>
    solveSeconds: number
}

/** (生命格点, 首领血量格点, 回复价格, 烟雾弹价格)；价格列出两个网格的全部组合 */
export function grids(config: MdpConfig): [Float64Array, Float64Array, Float64Array, Float64Array] {
    const n = config.hpPoints
    const grid = Float64Array.from({ length: n }, (_, i) => (E_MAX * i) / (n - 1))
    const nb = config.bossPoints
    const bossGrid = Float64Array.from({ length: nb }, (_, i) => (i === 0 ? 0.02 : i / (nb - 1)))
    const lam: number[] = []
    const mu: number[] = []
    for (const l of config.lambdas) {
        for (const m of config.smokePrices) {
            lam.push(l)
            mu.push(m)
        }
    }
    return [grid, bossGrid, Float64Array.from(lam), Float64Array.from(mu)]
}

/** 各价格在 floor、有效生命 e 处的值 */
export function tableAt(t: ValueTables, table: Float32Array, floor: number, e: number): Float64Array {
    const L = t.lambdas.length
    const N = t.grid.length
    const out = new Float64Array(L)
    if (!inRange(t, floor)) return out
    const { lo, w } = interp([e], t.grid)
    for (let l = 0; l < L; l++) {
        const base = (floor * L + l) * N + lo[0]
        out[l] = (1 - w[0]) * table[base] + w[0] * table[base + 1]
    }
    return out
}

/** 某价格下一层的值曲线（长度 = 生命格点数） */
export function tableRow(t: ValueTables, table: Float32Array, floor: number, li: number): Float64Array {
    const N = t.grid.length
    const L = t.lambdas.length
    if (!inRange(t, floor)) return new Float64Array(N)
    const base = (floor * L + li) * N
    return Float64Array.from(table.subarray(base, base + N))
}

export function inRange(t: ValueTables, floor: number): boolean {
    return floor >= 1 && floor < t.config.maxFloor
}

/** 首领第 window 个窗口前、有效生命 e、首领血量比例 b 时各价格的值 */
export function bossAt(t: ValueTables, floor: number, window: number, e: number, b: number): Float64Array {
    const L = t.lambdas.length
    const N = t.grid.length
    const NB = t.bossGrid.length
    const out = new Float64Array(L)
    const table = t.boss.get(floor)
    if (!table) return out
    const wdx = Math.min(window, BOSS_WINDOWS) - 1
    const ie = interp([e], t.grid)
    const ib = interp([b], t.bossGrid)
    const cells: [number, number, number][] = [
        [ie.lo[0], ib.lo[0], (1 - ie.w[0]) * (1 - ib.w[0])],
        [ie.lo[0], ib.lo[0] + 1, (1 - ie.w[0]) * ib.w[0]],
        [ie.lo[0] + 1, ib.lo[0], ie.w[0] * (1 - ib.w[0])],
        [ie.lo[0] + 1, ib.lo[0] + 1, ie.w[0] * ib.w[0]]
    ]
    for (let l = 0; l < L; l++) {
        let s = 0
        for (const [n, b2, wt] of cells) s += wt * table[((wdx * L + l) * N + n) * NB + b2]
        out[l] = s
    }
    return out
}

/** 首领表中某价格、某窗口的 [生命][首领血量] 切片 */
export function bossSlice(t: ValueTables, floor: number, window: number, li: number): Float64Array {
    const L = t.lambdas.length
    const N = t.grid.length
    const NB = t.bossGrid.length
    const table = t.boss.get(floor)
    const out = new Float64Array(N * NB)
    if (!table) return out
    const base = ((Math.min(window, BOSS_WINDOWS) - 1) * L + li) * N * NB
    out.set(table.subarray(base, base + N * NB))
    return out
}

/** 对偶界 V + λH + μS 最小的价格索引（回复单位 H，烟雾弹 S） */
export function priceIndex(t: ValueTables, values: ArrayLike<number>, heals: number, smokes = 0): number {
    let best = 0
    let bestV = Infinity
    for (let p = 0; p < values.length; p++) {
        const v = values[p] + t.lambdas[p] * Math.max(0, heals) + t.smokePrices[p] * Math.max(0, smokes)
        if (v < bestV) {
            bestV = v
            best = p
        }
    }
    return best
}

// ---------------------------------------------------------------------------- 转移核
function startHp(build: Build, e: number): [number, number] {
    return [Math.min(e, 1.0) * build.hp, Math.max(e - 1.0, 0.0) * build.hp]
}

/**
 * (kill, draw)：从 grid[i] 出发，打完一场普通战斗（敌人死亡 / 回合上限撤退）后落到 grid[j] 的概率 [i][j]。
 * kill + draw 每行之和为存活概率；第 0 行（已阵亡）为空。
 */
export function regularKernel(
    build: Build,
    foe: Foe,
    grid: Float64Array,
    samples: number,
    rng: Rng,
    rounds = REGULAR_ROUND_CAP
): { kill: Float64Array; draw: Float64Array } {
    const n = grid.length
    const count = (n - 1) * samples
    const start = new Int32Array(count)
    const hp = new Float64Array(count)
    const shield = new Float64Array(count)
    for (let k = 0; k < count; k++) {
        start[k] = 1 + Math.floor(k / samples)
        ;[hp[k], shield[k]] = startHp(build, grid[start[k]])
    }
    const res = fight(build, hp, shield, foe, { boss: false, rounds, rng })
    const killS: number[] = []
    const killX: number[] = []
    const drawS: number[] = []
    const drawX: number[] = []
    for (let k = 0; k < count; k++) {
        if (res.hp[k] <= 0) continue
        const e1 = (res.hp[k] + res.shield[k]) / build.hp
        if (res.ehp[k] <= 0) {
            killS.push(start[k])
            killX.push(e1)
        } else {
            drawS.push(start[k])
            drawX.push(e1)
        }
    }
    return { kill: deposit(killS, killX, grid, n, 1 / samples), draw: deposit(drawS, drawX, grid, n, 1 / samples) }
}

/**
 * 一段战斗（首领窗口、精英半场），从每个 (生命, 敌人血量) 状态出发；fresh 时只从满血敌人出发（行数 = 生命格点数）。
 * foe.hp 为敌人满血。返回 cont[s][s']：双方都存活、落到扁平状态 s' = 生命 × 敌人血量格点数 + 敌人血量；
 * kill[s][j]：敌人死亡、玩家落到生命格点 j（击杀回复已计入）。阵亡起点的行为空。
 */
export function segmentKernel(
    build: Build,
    foe: Foe,
    opts: {
        boss: boolean
        startRound: number
        rounds: number
        grid: Float64Array
        enemyGrid: Float64Array
        samples: number
        rng: Rng
        fresh?: boolean
    }
): { cont: Float64Array; kill: Float64Array; rows: number } {
    const { grid, enemyGrid, samples } = opts
    const n = grid.length
    const nb = enemyGrid.length
    const states = n * nb
    const rows = opts.fresh ? n : states
    const count = rows * samples
    const start = new Int32Array(count)
    const ei = new Int32Array(count)
    const hp = new Float64Array(count)
    const shield = new Float64Array(count)
    const ehp = new Float64Array(count)
    for (let k = 0; k < count; k++) {
        const s = Math.floor(k / samples)
        start[k] = s
        ei[k] = opts.fresh ? s : Math.floor(s / nb)
        const b0 = opts.fresh ? 1.0 : enemyGrid[s % nb]
        ;[hp[k], shield[k]] = startHp(build, grid[ei[k]])
        ehp[k] = b0 * foe.hp
    }
    const res = fight(build, hp, shield, foe, {
        boss: opts.boss,
        ehp,
        startRound: opts.startRound,
        rounds: opts.rounds,
        rng: opts.rng
    })
    const kill = new Float64Array(rows * n)
    const cont = new Float64Array(rows * states)
    const killS: number[] = []
    const killX: number[] = []
    const goK: number[] = []
    for (let k = 0; k < count; k++) {
        if (!(res.hp[k] > 0 && ei[k] > 0)) continue
        if (res.ehp[k] <= 0) {
            killS.push(start[k])
            killX.push((res.hp[k] + res.shield[k]) / build.hp)
        } else goK.push(k)
    }
    kill.set(deposit(killS, killX, grid, rows, 1 / samples))
    const ie = interp(
        goK.map(k => (res.hp[k] + res.shield[k]) / build.hp),
        grid
    )
    const ib = interp(
        goK.map(k => res.ehp[k] / foe.hp),
        enemyGrid
    )
    for (let q = 0; q < goK.length; q++) {
        const row = start[goK[q]] * states
        const e0 = ie.lo[q]
        const b0 = ib.lo[q]
        const we = ie.w[q]
        const wb = ib.w[q]
        cont[row + e0 * nb + b0] += ((1 - we) * (1 - wb)) / samples
        cont[row + e0 * nb + b0 + 1] += ((1 - we) * wb) / samples
        cont[row + (e0 + 1) * nb + b0] += (we * (1 - wb)) / samples
        cont[row + (e0 + 1) * nb + b0 + 1] += (we * wb) / samples
    }
    return { cont, kill, rows }
}

/** 精英整备后：从扁平状态 s 打完剩余回合并存活、落到生命格点 j 的概率（击杀与回合上限撤退都回到本层 ready） */
export function elitePauseKernel(
    build: Build,
    foe: Foe,
    grid: Float64Array,
    enemyGrid: Float64Array,
    samples: number,
    rng: Rng
): Float64Array {
    const n = grid.length
    const nb = enemyGrid.length
    const { cont, kill, rows } = segmentKernel(build, foe, {
        boss: false,
        startRound: ELITE_BREAK + 1,
        rounds: ELITE_ROUND_CAP - ELITE_BREAK,
        grid,
        enemyGrid,
        samples,
        rng
    })
    const lands = new Float64Array(rows * n)
    for (let s = 0; s < rows; s++) {
        for (let j = 0; j < n; j++) {
            let v = kill[s * n + j]
            for (let b = 0; b < nb; b++) v += cont[s * rows + j * nb + b]
            lands[s * n + j] = v
        }
    }
    return lands
}

/** 精英整备时的值 [价格][生命][精英血量]：任意回复后打完剩余回合。ready: [价格][生命] */
export function elitePauseValue(
    lands: Float64Array,
    ready: Float64Array,
    L: number,
    heal: Closure,
    nb: number,
    deathCost = 0
): Float64Array {
    const n = heal.size
    const s = (lands.length / n) | 0
    const values = mulBT(ready, L, n, lands, s)
    if (deathCost) {
        const death = deathOf(s, lands)
        for (let l = 0; l < L; l++) for (let r = 0; r < s; r++) values[l * s + r] -= deathCost * death[r]
    }
    return heal.apply(values, L, n, nb)
}

export function fleeChance(build: Build, floor: number): number {
    return Math.min(1.0, Math.min(0.9, Math.max(0.15, 0.45 + (build.agi - 1.5 * floor) * 0.015)) + build.fleeBonus)
}

/** 逃跑失败时那一下（不可闪避）的预期损失（最大生命比例） */
export function fleeHit(build: Build, foe: Foe): number {
    const raw = ((foe.atk * 100) / (100 + 2.5 * effDef(build))) * (1 - build.reduction) * build.takenMult
    return Math.max(1.0, raw) / build.hp
}

// ---------------------------------------------------------------------------- 回复闭包
/**
 * V* = max(V, max_i V*(apply_i(e)) − λ·cost_i)，对所有价格向量化。值数组布局 [价格][生命][其余维度]。
 * 回复的移动只连接相邻两个格点，按稀疏插值计算。
 */
export class Closure {
    readonly size: number
    private moves: Interp[]
    private costs: Float64Array[]

    constructor(
        items: HealItem[],
        grid: Float64Array,
        maxHp: number,
        lambdas: Float64Array,
        private readonly iterations: number
    ) {
        this.size = grid.length
        this.moves = items.map(it =>
            interp(
                Array.from(grid, e => healApply(it, e, maxHp)),
                grid
            )
        )
        this.costs = items.map(it => lambdas.map(l => healUnits(it, maxHp) * l))
    }

    apply(base: Float64Array, L: number, N: number, R = 1): Float64Array {
        if (!this.moves.length) return base
        let value = base
        for (let iter = 0; iter < this.iterations; iter++) {
            const next = Float64Array.from(base)
            for (let m = 0; m < this.moves.length; m++) {
                const { lo, w } = this.moves[m]
                const cost = this.costs[m]
                for (let l = 0; l < L; l++) {
                    const c = cost[l]
                    const plane = l * N * R
                    for (let i = 0; i < N; i++) {
                        const a = plane + lo[i] * R
                        const b = a + R
                        const wi = w[i]
                        const out = plane + i * R
                        for (let r = 0; r < R; r++) {
                            const cand = (1 - wi) * value[a + r] + wi * value[b + r] - c
                            if (cand > next[out + r]) next[out + r] = cand
                        }
                    }
                }
            }
            if (allClose(next, value)) return next
            value = next
        }
        return value
    }
}

/** numpy.allclose（rtol 1e-5，atol 1e-8） */
function allClose(a: Float64Array, b: Float64Array, atol = 1e-8): boolean {
    for (let k = 0; k < a.length; k++) if (Math.abs(a[k] - b[k]) > atol + 1e-5 * Math.abs(b[k])) return false
    return true
}

// ---------------------------------------------------------------------------- 求解
export type Progress = (message: string) => void

/** 从 maxFloor 逆向归纳到 F1，所有价格一并求解 */
export function solve(build: Build, mapId = 'mall', config: MdpConfig = DEFAULT_MDP, progress?: Progress): ValueTables {
    const steps = solveSteps(build, mapId, config, progress)
    for (;;) {
        const next = steps.next()
        if (next.done) return next.value
    }
}

/** 同 solve，每算完一层让出一次（调用方借此分片执行，不长时间阻塞页面） */
export function* solveSteps(
    build: Build,
    mapId = 'mall',
    config: MdpConfig = DEFAULT_MDP,
    progress?: Progress
): Generator<void, ValueTables, void> {
    const t0 = Date.now()
    const rng = seededRng(config.seed)
    const [grid, bossGrid, lambdas, smokePrices] = grids(config)
    const N = grid.length
    const NB = bossGrid.length
    const L = lambdas.length
    const top = config.maxFloor
    const maxHp = build.hp
    const heals = [...Object.values(SHOP_HEALS), ...Object.values(FIELD_HEALS)]
    const anyHeal = new Closure(heals, grid, maxHp, lambdas, config.healIterations)
    const calmHeal = new Closure(
        heals.filter(h => !h.combatOnly),
        grid,
        maxHp,
        lambdas,
        config.healIterations
    )
    const full = interp([1.0], grid)
    const low = Array.from(grid, e => e <= RUSH_STOP_HP)
    const LN = L * N
    const tableSize = (top + 2) * LN
    const ready = new Float32Array(tableSize)
    const step = new Float32Array(tableSize)
    const rush = new Float32Array(tableSize)
    const rushElite = new Float32Array(tableSize)
    const arrive = new Float32Array(tableSize)
    const encNormal = new Float32Array(tableSize)
    const encElite = new Float32Array(tableSize)
    const boss = new Map<number, Float32Array>()
    const bossStart = new Map<number, Float64Array>() // [价格][生命]：首领满血、第 1 窗口前
    const row = (table: Float32Array, g: number) => Float64Array.from(table.subarray(g * LN, (g + 1) * LN))
    // 连战中途进入 h 层（剩 j 层，不是第一层）的值：mid.get(h)[j-1]
    const zeroLayers = () => Array.from({ length: RUSH_FLOORS }, () => new Float64Array(LN))
    const mid = new Map<number, Float64Array[]>([
        [top + 1, zeroLayers()],
        [top + 2, zeroLayers()]
    ])
    const midElite = new Map<number, Float64Array[]>([
        [top + 1, zeroLayers()],
        [top + 2, zeroLayers()]
    ]) // 同上，开启「自动打精英」
    // 在 h 层从满血精英打到整备的值（回复后打完剩余回合）：[价格][出发生命]
    const pauseReached = new Map<number, Float64Array>()

    const kernels = new Map<string, { kill: Float64Array; draw: Float64Array }>()
    const eliteKernels = new Map<number, { cont: Float64Array; kill: Float64Array; lands: Float64Array }>()
    const reference = (floor: number) =>
        Math.min(top, Math.floor((floor - 1) / config.kernelStride) * config.kernelStride + config.kernelStride)
    const kernel = (kind: EnemyKind, floor: number) => {
        const ref = reference(floor)
        const key = `${kind}|${ref}`
        let k = kernels.get(key)
        if (!k) {
            const rounds = kind === 'elite' ? ELITE_ROUND_CAP : REGULAR_ROUND_CAP
            k = regularKernel(build, foeAt(kind, ref, mapId, build), grid, config.samples, rng, rounds)
            kernels.set(key, k)
        }
        return k
    }
    const eliteKernel = (floor: number) => {
        const ref = reference(floor)
        let k = eliteKernels.get(ref)
        if (!k) {
            const foe = foeAt('elite', ref, mapId, build)
            const first = segmentKernel(build, foe, {
                boss: false,
                startRound: 1,
                rounds: ELITE_BREAK,
                grid,
                enemyGrid: bossGrid,
                samples: config.samples,
                rng,
                fresh: true
            })
            const lands = elitePauseKernel(build, foe, grid, bossGrid, config.pauseSamples, rng)
            k = { cont: first.cont, kill: first.kill, lands }
            eliteKernels.set(ref, k)
        }
        return k
    }
    const [gain, risk] = lootProfile(config, mapId, top)
    const gold = config.gold
    const income = (floor: number) => {
        const units = fieldHealIncome(floor, maxHp)
        return (l: number) => lambdas[l] * units + gain[floor]
    }
    /** 转移核的死亡概率（每个出发生命格点），按转移核缓存 */
    const deaths = new WeakMap<Float64Array, Float64Array>()
    const deathFrom = (...parts: Float64Array[]): Float64Array => {
        let d = deaths.get(parts[0])
        if (!d) {
            d = deathOf(N, ...parts)
            deaths.set(parts[0], d)
        }
        return d
    }
    /** 金币目标：撤离（再值 0，带回背包）与已阵亡状态（损失背包） */
    const bank = (values: Float64Array, h: number): Float64Array => {
        if (!gold) return values
        for (let l = 0; l < L; l++) {
            for (let i = 0; i < N; i++) values[l * N + i] = i === 0 ? -risk[h] : Math.max(values[l * N + i], 0)
        }
        return values
    }
    /** 连战中进入 h 层的值；after = 本层处理完后的延续。autoElite：HP 高于 50% 时连战自己打精英到整备，击杀则继续 */
    const enter = (h: number, after: Float64Array, stopLow: boolean, autoElite = false): Float64Array => {
        const inc = income(h)
        const out = new Float64Array(LN)
        if (isBossFloor(h)) {
            const bs = bossStart.get(h)!
            for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) out[l * N + i] = inc(l) + bs[l * N + i]
            return out
        }
        const mix = floorMix(h)
        const { kill, draw } = kernel('normal', h)
        const readyH = row(ready, h)
        const fought = mulBT(after, L, N, kill, N)
        const drawn = mulBT(readyH, L, N, draw, N)
        const dNormal = gold ? deathFrom(kill, draw) : null
        const eN = row(encNormal, h)
        const eE = row(encElite, h)
        let auto: Float64Array | null = null
        if (autoElite) {
            const first = eliteKernel(h)
            auto = mulBT(after, L, N, first.kill, N)
            const reached = pauseReached.get(h)!
            const dFirst = gold ? deathFrom(first.cont, first.kill) : null
            for (let k = 0; k < LN; k++) auto[k] += reached[k] - (dFirst ? risk[h] * dFirst[k % N] : 0)
        }
        for (let l = 0; l < L; l++) {
            for (let i = 0; i < N; i++) {
                const k = l * N + i
                const f = fought[k] + drawn[k] - (dNormal ? risk[h] * dNormal[i] : 0) // 平局让连战停在该层
                const normal = stopLow && low[i] ? eN[k] : f
                const elite = auto && !low[i] ? auto[k] : eE[k]
                out[k] = inc(l) + mix.normal * normal + mix.elite * elite + mix.choice * readyH[k] + mix.loot * after[k]
            }
        }
        return out
    }

    for (let g = top; g >= 1; g--) {
        const off = g * LN
        // 1. 从 g 层 ready 状态发起连战（第一层不做下潜前的 HP 检查）
        if (g + 1 <= top) rush.set(enter(g + 1, mid.get(g + 2)![RUSH_FLOORS - 2], true), off)
        if (config.rushElites && g + 1 <= top) {
            rushElite.set(enter(g + 1, midElite.get(g + 2)![RUSH_FLOORS - 2], true, true), off)
        }
        // 2. g 层 ready：单步、连战（两种）或先回复
        step.set(arrive.subarray((g + 1) * LN, (g + 2) * LN), off)
        const best = new Float64Array(LN)
        for (let k = 0; k < LN; k++) best[k] = Math.max(step[off + k], rush[off + k], rushElite[off + k])
        ready.set(bank(calmHeal.apply(best, L, N), g), off)
        const readyG = row(ready, g)
        // 3. g 层的遭遇
        for (const kind of ['normal', 'elite'] as const) {
            const foe = foeAt(kind, g, mapId, build)
            let fightValue: Float64Array
            if (kind === 'elite') {
                const { cont, kill, lands } = eliteKernel(g)
                const pause = elitePauseValue(lands, readyG, L, anyHeal, NB, risk[g])
                if (gold) {
                    // 在整备时撤离；已阵亡状态损失背包
                    for (let k = 0; k < pause.length; k++) {
                        pause[k] = Math.floor(k / NB) % N === 0 ? -risk[g] : Math.max(pause[k], 0)
                    }
                }
                fightValue = mulBT(readyG, L, N, kill, N)
                const reached = mulBT(pause, L, N * NB, cont, N)
                pauseReached.set(g, reached)
                const dFirst = gold ? deathFrom(cont, kill) : null
                for (let k = 0; k < LN; k++) fightValue[k] += reached[k] - (dFirst ? risk[g] * dFirst[k % N] : 0)
            } else {
                const { kill, draw } = kernel(kind, g)
                const both = new Float64Array(kill.length)
                for (let k = 0; k < both.length; k++) both[k] = kill[k] + draw[k]
                fightValue = mulBT(readyG, L, N, both, N)
                const d = gold ? deathFrom(kill, draw) : null
                if (d) for (let k = 0; k < LN; k++) fightValue[k] -= risk[g] * d[k % N]
            }
            const failAt = interp(
                Array.from(grid, e => Math.max(e - fleeHit(build, foe), 0)),
                grid
            )
            const p = fleeChance(build, g)
            const value = new Float64Array(LN)
            for (let l = 0; l < L; l++) {
                for (let i = 0; i < N; i++) {
                    const k = l * N + i
                    if (i === 0) continue // 已阵亡
                    const failed =
                        (1 - failAt.w[i]) * fightValue[l * N + failAt.lo[i]] +
                        failAt.w[i] * fightValue[l * N + failAt.lo[i] + 1]
                    const flee = p * readyG[k] + (1 - p) * failed
                    const smoke = readyG[k] - smokePrices[l] // 烟雾弹：无伤脱身，留在本层
                    value[k] = Math.max(fightValue[k], flee, smoke)
                }
                value[l * N] = Math.max(fightValue[l * N], -risk[g])
            }
            ;(kind === 'normal' ? encNormal : encElite).set(bank(anyHeal.apply(value, L, N), g), off)
        }
        // 4. g 层的首领
        if (isBossFloor(g)) {
            const win = new Float64Array(L)
            for (let l = 0; l < L; l++) {
                const k = l * N + full.lo[0]
                win[l] = config.bossValue + (1 - full.w[0]) * readyG[k] + full.w[0] * readyG[k + 1]
            }
            const table = yield* solveBoss(
                build,
                foeAt('boss', g, mapId, build),
                grid,
                bossGrid,
                lambdas,
                win,
                anyHeal,
                config,
                rng,
                risk[g]
            )
            boss.set(g, table)
            const bs = new Float64Array(LN)
            for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) bs[l * N + i] = table[(l * N + i) * NB + NB - 1]
            bossStart.set(g, bs)
        }
        // 5. 单步到达 g 层
        const inc = income(g)
        if (isBossFloor(g)) {
            const bs = bossStart.get(g)!
            for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) arrive[off + l * N + i] = inc(l) + bs[l * N + i]
        } else {
            const mix = floorMix(g)
            for (let l = 0; l < L; l++) {
                for (let i = 0; i < N; i++) {
                    const k = l * N + i
                    arrive[off + k] =
                        inc(l) +
                        mix.normal * encNormal[off + k] +
                        mix.elite * encElite[off + k] +
                        (mix.choice + mix.loot) * readyG[k]
                }
            }
        }
        if (gold) for (let l = 0; l < L; l++) arrive[off + l * N] = -risk[g]
        // 6. 连战中途进入 g+1 层（需要 ready[g] 做下潜前的 HP 检查）
        if (g + 1 <= top) {
            const midLayers = (layers: Map<number, Float64Array[]>, autoElite: boolean) => {
                const out: Float64Array[] = []
                for (let j = 1; j < RUSH_FLOORS; j++) {
                    const after = j === 1 ? row(ready, g + 1) : layers.get(g + 2)![j - 2]
                    const entered = enter(g + 1, after, true, autoElite)
                    for (let l = 0; l < L; l++)
                        for (let i = 0; i < N; i++) if (low[i]) entered[l * N + i] = readyG[l * N + i]
                    out.push(entered)
                }
                out.push(new Float64Array(LN))
                layers.set(g + 1, out)
                layers.delete(g + 3)
            }
            midLayers(mid, false)
            if (config.rushElites) midLayers(midElite, true)
            pauseReached.delete(g + 2)
        }
        // 转移核只在逆推经过其区块时用到（本层与连战进入的上一层）：之后释放，F4000 时可省下数百 MB
        const keep = reference(g)
        for (const key of kernels.keys()) if (Number(key.split('|')[1]) > keep) kernels.delete(key)
        for (const ref of eliteKernels.keys()) if (ref > keep) eliteKernels.delete(ref)
        if (progress && g % 200 === 0) progress(`mdp: floor ${g} (${((Date.now() - t0) / 1000).toFixed(0)}s)`)
        yield
    }
    return {
        config,
        build,
        mapId,
        grid,
        bossGrid,
        lambdas,
        smokePrices,
        ready,
        step,
        rush,
        rushElite,
        encNormal,
        encElite,
        boss,
        solveSeconds: (Date.now() - t0) / 1000
    }
}

/**
 * 每个窗口前的值 [窗口][价格][生命][首领血量]；击杀价值为 bossValue + 满血 ready。每个转移核与每次迭代后让出。
 * 金币目标：死亡损失 deathCost，首领战中撤离再值 0
 */
function* solveBoss(
    build: Build,
    foe: Foe,
    grid: Float64Array,
    bossGrid: Float64Array,
    lambdas: Float64Array,
    winValue: Float64Array,
    heal: Closure,
    config: MdpConfig,
    rng: Rng,
    deathCost = 0
): Generator<void, Float32Array, void> {
    const N = grid.length
    const NB = bossGrid.length
    const L = lambdas.length
    const S = N * NB
    const kernels: { cont: Csr; win: Float64Array; death: Float64Array }[] = []
    for (let w = 0; w < BOSS_WINDOWS; w++) {
        const { cont, kill } = segmentKernel(build, foe, {
            boss: true,
            startRound: w * BOSS_WINDOW + 1,
            rounds: BOSS_WINDOW,
            grid,
            enemyGrid: bossGrid,
            samples: config.samples,
            rng
        })
        const win = new Float64Array(S)
        for (let s = 0; s < S; s++) for (let j = 0; j < N; j++) win[s] += kill[s * N + j]
        const death = deathOf(S, cont, kill) // 金币目标的死亡概率（每个出发状态）
        kernels.push({ cont: toCsr(cont, S, S), win, death }) // 每行只落到少数状态：稀疏乘法快数倍
        yield
    }
    const windowValue = (w: number, next: Float64Array): Float64Array => {
        const { cont, win, death } = kernels[w]
        const v = mulBTSparse(next, L, S, cont)
        for (let l = 0; l < L; l++) {
            for (let s = 0; s < S; s++) v[l * S + s] += winValue[l] * win[s] - (config.gold ? deathCost * death[s] : 0)
        }
        const out = heal.apply(v, L, N, NB)
        if (config.gold) {
            for (let k = 0; k < out.length; k++)
                out[k] = Math.floor(k / NB) % N === 0 ? -deathCost : Math.max(out[k], 0)
        }
        return out
    }
    const values = new Float32Array(BOSS_WINDOWS * L * S)
    let last: Float64Array = new Float64Array(L * S)
    for (let it = 0; it < config.bossIterations; it++) {
        const next = windowValue(BOSS_WINDOWS - 1, last)
        if (allClose(next, last, 1e-6)) {
            last = next
            break
        }
        last = next
        yield
    }
    values.set(last, (BOSS_WINDOWS - 1) * L * S)
    let nextWindow: Float64Array = last
    for (let w = BOSS_WINDOWS - 2; w >= 0; w--) {
        nextWindow = windowValue(w, nextWindow)
        values.set(nextWindow, w * L * S)
    }
    return values
}

/** 与 combat 估算相同的在线校准 */
export function withCalibration(build: Build, kPlayer: number, kEnemy: number): Build {
    return { ...build, dealtMult: build.dealtMult * kPlayer, takenMult: build.takenMult * kEnemy }
}
