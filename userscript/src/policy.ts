// ============================================================================
// policy.ts — 刷首领决策（移植自 policy.py）
//
// 目标：每张门票击杀尽可能多的首领。彩虹掉落直接进仓库，背包是次要的；
// 深度（= 首领数）才是关键。decide(state) 是当前状态的纯函数，不修改状态。
// ============================================================================

import { chainRisk, estimate, runBuild } from './combat.ts'
import type { Calibration, FightEstimate } from './combat.ts'
import { fixed, int, pct } from './format.ts'
import { GearModel } from './gear.ts'
import type { Build } from './model.ts'
import type { Json } from './types.ts'

/** 商店补给的回复量（最大生命比例）；野外道具从描述中解析 */
export const SUPPLY_HEAL: Record<string, number> = {
    coffee: 0.4,
    ice_tea: 0.25,
    triage_vial: 0.57,
    barrier_serum: 0.5,
    slow_serum: 0.38,
    critical_serum: 0.55,
    field_surgery: 1.0
}
export const COMBAT_ONLY = new Set(['field_surgery'])
/** 服务器同时最多允许三个限层增益 */
const MAX_DURATION_BUFFS = 3
/** 局内要用的补给留在背包，永不锁入安全箱 */
export const KEEP_IN_BAG = new Set([
    ...Object.keys(SUPPLY_HEAL),
    'adrenaline',
    'grenade',
    'smoke',
    'shield',
    'flare',
    'pouch',
    'doll'
])
const SUPPLY_COST: Record<string, number> = {
    coffee: 300,
    ice_tea: 150,
    triage_vial: 780,
    barrier_serum: 1450,
    slow_serum: 1250,
    critical_serum: 2300,
    field_surgery: 3300
}
/** 对战斗有帮助的野外道具效果关键词（高风险战斗前使用） */
const COMBAT_BUFF_WORDS = [
    '攻击+',
    '防御+',
    '减伤',
    '闪避率+',
    '吸血',
    '首回合',
    '对首领',
    '暴击率+',
    '穿透+',
    '伤害 ×',
    '护盾'
]
/** 按描述给可佩戴临装打分的权重（按插入顺序匹配后缀） */
const WEAR_WEIGHTS: [string, number][] = [
    ['攻击', 1.0],
    ['防御', 0.7],
    ['减伤', 1.3],
    ['闪避率', 1.5],
    ['吸血', 1.4],
    ['对首领伤害', 0.8],
    ['暴击率', 0.6],
    ['穿透', 0.6],
    ['追击率', 0.4],
    ['敏捷', 0.25],
    ['首回合攻击', 0.2],
    ['逃跑率', 0.0]
]

export interface PolicyConfig {
    /** 普通战斗可接受的死亡概率 */
    maxDeathRisk: number
    /** 一个首领窗口内可接受的死亡概率 */
    maxBossWindowRisk: number
    /** 资源耗尽且高于此风险：撤离而不是硬打 */
    abortRisk: number
    /** 回复手段全部用完：在下一个安全时机撤离 */
    extractWhenDry: boolean
    /** 背包里有替身娃娃（死亡时保住本局与背包）：继续深入刷首领，不撤离 */
    greedyWithDoll: boolean
    /** 低于此 HP 比例：回血或逐层前进 */
    rushMinHp: number
    /** 可选的硬性上限（到达即撤离） */
    maxFloor: number | null
    /** 策划的平衡性调整（造成 ×2 / 承受 ×2）的评分；< 0 = 永不佩戴 */
    balanceGearScore: number
    samples: number
    /** 首层使用的暴走肾上腺素数量（最多 2 层） */
    startAdrenaline: number
    /** 连战允许的最大死亡概率（连续 4 场普通战斗） */
    maxRushRisk: number
    /** 安全箱满时，战利品价值 ≥ 倍数 × 箱内最便宜物品才替换 */
    swapMargin: number
    /** 开局使用哆啦的临时扩容袋（安全箱 +1 格）；扩容袋珍贵，默认不用（捡到的带回家） */
    usePouch: boolean
    /** 只剩战斗专用回复时，HP 低于此比例才在遭遇中使用 */
    combatTopupHp: number
    /** 遗物盒估值 ≥ 此值时舔包，否则捞尸换赏金；舔到的物资仍有这么多放不进安全箱时撤离带回 */
    relicLootValue: number
}

export const DEFAULT_POLICY: PolicyConfig = {
    maxDeathRisk: 0.02,
    maxBossWindowRisk: 0.03,
    abortRisk: 0.1,
    extractWhenDry: true,
    greedyWithDoll: true,
    rushMinHp: 0.55,
    maxFloor: null,
    balanceGearScore: -1.0,
    samples: 600,
    startAdrenaline: 2,
    maxRushRisk: 0.01,
    swapMargin: 1.2,
    usePouch: false,
    combatTopupHp: 0.3,
    relicLootValue: 10_000_000
}

export interface Decision {
    command: string
    params: Record<string, unknown>
    reason: string
    estimate?: FightEstimate
}

export const decision = (
    command: string,
    params: Record<string, unknown>,
    reason: string,
    est?: FightEstimate
): Decision => ({
    command,
    params,
    reason,
    estimate: est
})

function pctOf(text: string, pattern: RegExp): number {
    const m = pattern.exec(text)
    return m ? Number(m[1]) / 100 : 0
}

export function controlsOf(state: Json): Map<string, Json> {
    return new Map(((state.controls || []) as Json[]).map(c => [c.command, c]))
}

/** 参数的稳定键（与 Python 的 repr(sorted(params.items())) 作用相同） */
export function paramsKey(params: Record<string, unknown>): string {
    return JSON.stringify(Object.entries(params).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
}

/** 与 Python max(key=...) 一致：相等时取第一个 */
function maxBy<T>(items: T[], key: (x: T) => number | number[]): T {
    let best = items[0]
    let bestKey = key(best)
    for (const it of items.slice(1)) {
        const k = key(it)
        if (compareKeys(k, bestKey) > 0) {
            best = it
            bestKey = k
        }
    }
    return best
}

function minBy<T>(items: T[], key: (x: T) => number): T {
    return maxBy(items, x => -key(x))
}

function compareKeys(a: number | number[], b: number | number[]): number {
    const aa = Array.isArray(a) ? a : [a]
    const bb = Array.isArray(b) ? b : [b]
    for (let i = 0; i < aa.length; i++) if (aa[i] !== bb[i]) return aa[i] > bb[i] ? 1 : -1
    return 0
}

type Heal = [number, Json]

const RELIC_VALUE = /估值\s*([\d,]+)\s*金/

/** 遗物盒估值（事件文本「……遗落的物资，估值 1,234 金。」） */
export function relicValue(event: Json): number {
    const m = RELIC_VALUE.exec(`${event.text || ''} ${event.description || ''}`)
    return m ? Number(m[1].replaceAll(',', '')) : 0
}

export class BossFarmPolicy {
    readonly cfg: PolicyConfig
    /** 服务器拒绝过的 (楼层, 命令, 参数)；该楼层不再提出 */
    protected banned = new Set<string>()
    /** 本局服务器提示奇遇攻击加成已达上限（+60%） */
    eventAtkCapped = false
    adrenalineUsed = 0
    pouchUsed = false
    /** 最近一次没有限层增益时看到的实时属性（含减伤与当时穿戴的临装） */
    baseStats: Json = null
    /** 道具 → 仍被禁用到的楼层（如缓释再生液还在生效时每支都会被拒绝） */
    private blockedUntil = new Map<string, number>()
    /** 舔到的遗物盒仍有部分不在安全箱里 */
    banking = false
    affixBuild: Build | null = null
    loadoutKey: string | null = null

    constructor(
        readonly cal: Calibration,
        config: Partial<PolicyConfig> = {}
    ) {
        this.cfg = { ...DEFAULT_POLICY, ...config }
    }

    newRun(): void {
        this.banned.clear()
        this.eventAtkCapped = false
        this.adrenalineUsed = 0
        this.pouchUsed = false
        this.baseStats = null
        this.blockedUntil.clear()
        this.banking = false
    }

    /** 响应服务器的限制提示 */
    note(notices: string[]): void {
        for (const text of notices) {
            if (text.includes('奇遇攻击加成已达')) this.eventAtkCapped = true
            if (text.includes('扩容袋')) this.pouchUsed = true
        }
    }

    /** 服务器拒绝了 d：本层不再提出；itemId / message 为被拒绝的道具及原因 */
    ban(floor: number, d: Decision, itemId: string | null = null, message = ''): void {
        this.banned.add(`${floor}|${d.command}|${paramsKey(d.params)}`)
        if (itemId && message.includes('还在生效')) this.blockedUntil.set(itemId, floor + 3)
        // 攻击加成已达 +50%（例如续玩的对局）：本局不再打肾上腺素
        if (itemId === 'adrenaline') this.adrenalineUsed = this.cfg.startAdrenaline
    }

    protected itemBlocked(run: Json, itemId: string): boolean {
        return run.floor <= (this.blockedUntil.get(itemId) ?? -1)
    }

    protected allowed(floor: number, command: string, params: Record<string, unknown>): boolean {
        return !this.banned.has(`${floor}|${command}|${paramsKey(params)}`)
    }

    // ------------------------------------------------------------------ 辅助
    /** 实时战斗 Build；永久词条按当前穿戴只推导一次 */
    currentBuild(state: Json): Build {
        const loadout = JSON.stringify(Object.entries(state.player?.loadout || {}).sort())
        if (this.affixBuild === null || loadout !== this.loadoutKey) {
            const gm = new GearModel(state)
            this.affixBuild = gm.build(gm.current)
            this.loadoutKey = loadout
        }
        return runBuild(state, this.affixBuild)
    }

    /**
     * 不含限层增益的 Build（连战跨越多层时增益可能中途失效）。
     * 使用最近一次没有此类增益时观测到的实时属性。
     */
    lastingBuild(state: Json): Build {
        const run = state.run
        const build = this.currentBuild(state)
        const worn = JSON.stringify(((run.field_slots || []) as Json[]).map(slot => slot?.id ?? null))
        if (!(run.field_buffs || []).length)
            this.baseStats = { ...run.effective_stats, reduction: build.reduction, worn }
        const base = this.baseStats
        if (isEmpty(base)) return build
        if (base.worn === worn) {
            // 临装未变：增益也可能降低属性（如清仓红牌防御 -20%），直接用无增益时的数值
            return { ...build, atk: base.atk, defense: base.def, agi: base.agi, reduction: base.reduction }
        }
        return {
            ...build,
            atk: Math.min(build.atk, base.atk),
            defense: Math.min(build.defense, base.def),
            agi: Math.min(build.agi, base.agi),
            reduction: Math.min(build.reduction, base.reduction)
        }
    }

    estimate(state: Json): FightEstimate {
        return estimate(this.currentBuild(state), state, this.cal, this.cfg.samples)
    }

    private static fieldHeals(run: Json): Heal[] {
        const out: Heal[] = []
        for (const item of (run.field || []) as Json[]) {
            if (item.healing && (item.quantity ?? 0) > 0)
                out.push([pctOf(item.description || '', /回复 (\d+)% 最大生命/), item])
        }
        return out
    }

    private static bagHeals(run: Json, inCombat: boolean): Heal[] {
        const out: Heal[] = []
        for (const item of (run.bag || []) as Json[]) {
            const iid = item.id
            if (iid in SUPPLY_HEAL && (inCombat || !COMBAT_ONLY.has(iid))) {
                let heal = SUPPLY_HEAL[iid]
                if (iid === 'critical_serum' && run.hp <= 0.35 * run.max_hp) heal = 0.9
                out.push([heal, item])
            }
        }
        return out
    }

    private static activeBuffIds(run: Json): Set<string> {
        const ids = new Set<string>()
        for (const entry of [...(run.field_buffs || []), ...(run.field_charges || [])]) {
            ids.add(entry !== null && typeof entry === 'object' ? entry.id : String(entry))
        }
        return ids
    }

    /** 选浪费最少的回复：先用免费野外道具，再用每点 HP 最便宜的补给 */
    private bestHeal(run: Json, inCombat: boolean, force: boolean): Decision | null {
        const deficit = 1 - run.hp / run.max_hp
        if (deficit <= 0.02) return null
        const floor = run.floor
        const fieldHeals = BossFarmPolicy.fieldHeals(run).filter(([, it]) =>
            this.allowed(floor, 'field_use', { id: it.id })
        )
        const fitting = fieldHeals.filter(([h]) => h <= deficit + 0.05)
        if (fitting.length || (force && fieldHeals.length)) {
            const [heal, item] = fitting.length ? maxBy(fitting, x => x[0]) : minBy(fieldHeals, x => x[0])
            return decision(
                'field_use',
                { id: item.id },
                `野外回复 ${item.name}（+${pct(heal)}，缺口 ${pct(deficit)}）`
            )
        }
        const bag = BossFarmPolicy.bagHeals(run, inCombat).filter(
            ([, it]) => this.allowed(floor, 'use', { index: it.index }) && !this.itemBlocked(run, it.id)
        )
        if (!bag.length) return null
        const fittingBag = bag.filter(([h]) => h <= deficit + 0.1)
        let chosen: Heal
        if (fittingBag.length) chosen = maxBy(fittingBag, x => [x[0], -(SUPPLY_COST[x[1].id] ?? 0)])
        else if (force) chosen = minBy(bag, x => SUPPLY_COST[x[1].id] ?? 0)
        else return null
        const [heal, item] = chosen
        return decision('use', { index: item.index }, `使用 ${item.name}（+${pct(heal)}，缺口 ${pct(deficit)}）`)
    }

    /**
     * HP 低且只剩战斗专用回复（战地手术匣）：在遭遇中立即使用，
     * 以便恢复连战，而不是在几场简单战斗里慢慢掉血。
     */
    private combatOnlyTopup(run: Json): Decision | null {
        if (run.hp >= this.cfg.combatTopupHp * run.max_hp || BossFarmPolicy.bagHeals(run, false).length) return null
        const floor = run.floor
        const kits = BossFarmPolicy.bagHeals(run, true)
            .map(([, it]) => it)
            .filter(it => COMBAT_ONLY.has(it.id) && this.allowed(floor, 'use', { index: it.index }))
        if (!kits.length) return null
        return decision(
            'use',
            { index: kits[0].index },
            `HP ${pct(run.hp / run.max_hp)} 且只剩战斗专用回复 → ${kits[0].name}`
        )
    }

    private combatBuff(run: Json, boss: boolean): Decision | null {
        const active = BossFarmPolicy.activeBuffIds(run)
        const durationSlotsFull = (run.field_buffs || []).length >= MAX_DURATION_BUFFS
        for (const item of (run.field || []) as Json[]) {
            const desc: string = item.description || ''
            if (item.healing || (item.quantity ?? 0) <= 0 || active.has(item.id)) continue
            if ((durationSlotsFull && desc.includes('持续')) || !this.allowed(run.floor, 'field_use', { id: item.id }))
                continue
            if (desc.includes('对首领') && !boss) continue
            if (COMBAT_BUFF_WORDS.some(word => desc.includes(word))) {
                return decision('field_use', { id: item.id }, `增益 ${item.name}：${desc}`)
            }
        }
        return null
    }

    protected static bagItem(run: Json, itemId: string): Json | null {
        return ((run.bag || []) as Json[]).find(it => it.id === itemId) ?? null
    }

    // ------------------------------------------------------------------ 决策
    /** 服务器在本层尚未拒绝过的最佳决策 */
    decide(state: Json): Decision {
        const d = this.decideRaw(state)
        const floor = state.run.floor
        if (this.allowed(floor, d.command, d.params)) return d
        const ctr = controlsOf(state)
        const foe = (state.run.event || {}).enemy
        for (const command of [...(foe ? ['fight'] : []), 'rush', 'down', 'leave', 'extract']) {
            if (ctr.has(command) && this.allowed(floor, command, {})) {
                return decision(command, {}, `${d.command} 被拒绝后的备选`)
            }
        }
        throw new AgentError(`第 ${floor} 层所有操作都被拒绝`)
    }

    protected decideRaw(state: Json): Decision {
        const run = state.run
        const ctr = controlsOf(state)
        if (!isEmpty(run.field_offer)) return this.fieldOffer(run)
        const boost = this.startBoost(run)
        if (boost) return boost
        const protect = this.protectCollectible(state, ctr) ?? this.secureLoot(state)
        if (protect) return protect
        const event = run.event || {}
        if (event.enemy && ctr.has('fight')) return this.fightDecision(state, ctr)
        if (event.kind === 'choice' && ((state.controls || []) as Json[]).some(c => c.command === 'event')) {
            return this.choice(state)
        }
        if (event.kind === 'relic' && (ctr.has('rob') || ctr.has('rescue'))) return this.relic(event, ctr)
        const bank = this.bankLoot(run, ctr)
        if (bank) return bank
        if (ctr.has('down') || ctr.has('rush')) return this.advance(state, ctr)
        if (ctr.has('leave')) return decision('leave', {}, '无可执行操作')
        if (ctr.has('extract')) return decision('extract', {}, '只能撤离')
        throw new AgentError(`没有已知的操作：${[...ctr.keys()].join(', ')}`)
    }

    /** 遇难玩家的遗物盒：估值 ≥ relicLootValue 时舔包，否则捞尸（赏金立即入账，无需带出） */
    private relic(event: Json, ctr: Map<string, Json>): Decision {
        const value = relicValue(event)
        if (value >= this.cfg.relicLootValue && ctr.has('rob')) {
            this.banking = true
            return decision(
                'rob',
                ctr.get('rob').params || {},
                `舔包：遗物估值 ${int(value)}（≥ ${int(this.cfg.relicLootValue)}）`
            )
        }
        if (ctr.has('rescue')) {
            return decision('rescue', ctr.get('rescue').params || {}, `捞尸：归还遗物（估值 ${int(value)}）换赏金`)
        }
        return decision('leave', {}, '遗物盒：无可执行操作')
    }

    /**
     * 舔包之后：仍有 ≥ relicLootValue 的物资在安全箱外（死亡会丢失）时撤离带回。在安全箱整理之后判断，
     * 只计放不进去的部分；全部放下后照常前进。普通战利品不会触发撤离。
     */
    private bankLoot(run: Json, ctr: Map<string, Json>): Decision | null {
        if (!this.banking) return null
        const exposed = ((run.bag || []) as Json[])
            .filter(it => it.can_protect && !KEEP_IN_BAG.has(it.id))
            .reduce((sum, it) => sum + Number(it.value || 0) * (it.quantity || 1), 0)
        if (exposed < this.cfg.relicLootValue) {
            this.banking = false
            return null
        }
        if ((run.event || {}).enemy || !ctr.has('extract')) return null
        return decision('extract', {}, `撤离：舔到的遗物仍有 ${int(exposed)} 放不进安全箱`)
    }

    /** 开局使用携带的肾上腺素（每支本局攻击 +25%，最多 2 支）；同时立即打开扩容袋（+1 格） */
    private startBoost(run: Json): Decision | null {
        if ((run.event || {}).enemy) return null
        const pouch = this.cfg.usePouch && !this.pouchUsed ? BossFarmPolicy.bagItem(run, 'pouch') : null
        if (pouch && this.allowed(run.floor, 'use', { index: pouch.index })) {
            this.pouchUsed = true // 每局一个；多余的留在背包
            return decision('use', { index: pouch.index }, '打开扩容袋（安全箱 +1 格）')
        }
        if (this.adrenalineUsed >= this.cfg.startAdrenaline) return null
        const shot = BossFarmPolicy.bagItem(run, 'adrenaline')
        if (!shot || run.hp < 0.5 * run.max_hp || !this.allowed(run.floor, 'use', { index: shot.index })) return null
        this.adrenalineUsed += 1
        return decision('use', { index: shot.index }, `肾上腺素 ${this.adrenalineUsed}/${this.cfg.startAdrenaline}`)
    }

    static missingCollectibles(state: Json): Set<string> {
        const atlas = state.atlas || {}
        return new Set(
            ((atlas.items || []) as Json[]).filter(i => !i.unlocked && i.rarity !== 'rainbow').map(i => i.id)
        )
    }

    /** 把仍缺的图鉴物品锁入安全箱（死亡也保留） */
    private protectCollectible(state: Json, ctr: Map<string, Json>): Decision | null {
        const run = state.run
        const missing = BossFarmPolicy.missingCollectibles(state)
        const secured = new Set(((run.safe || []) as Json[]).map(it => it.id))
        const pending = ((run.bag || []) as Json[]).filter(it => missing.has(it.id) && !secured.has(it.id))
        if (!pending.length) return null
        const item = pending[0]
        if (item.can_protect && (run.safe || []).length < (run.safe_capacity || 0)) {
            return decision('save_box', { index: item.index }, `锁入仍缺的图鉴物品 ${item.name}`)
        }
        if (ctr.has('extract') && !(run.event || {}).enemy) {
            return decision('extract', {}, `安全箱已满，撤离以带回 ${item.name}`)
        }
        return null
    }

    /** 把最值钱的战利品放进安全箱（失败也保留） */
    private secureLoot(state: Json): Decision | null {
        const run = state.run
        if ((run.event || {}).enemy) return null
        const missing = BossFarmPolicy.missingCollectibles(state)
        const floor = run.floor
        const worth = (item: Json): number =>
            missing.has(item.id) ? Infinity : Number(item.value || 0) * (item.quantity || 1)

        const candidates = ((run.bag || []) as Json[]).filter(
            it => it.can_protect && !KEEP_IN_BAG.has(it.id) && worth(it) > 0
        )
        if (!candidates.length) return null
        const best = maxBy(candidates, worth)
        const safe = (run.safe || []) as Json[]
        if (safe.length < (run.safe_capacity || 0)) {
            const params = { index: best.index }
            if (this.allowed(floor, 'save_box', params)) {
                return decision('save_box', params, `锁入 ${best.name}（${int(worth(best))}）`)
            }
            return null
        }
        if (!safe.length) return null
        const cheapestPos = minBy(
            safe.map((_, i) => i),
            i => worth(safe[i])
        )
        const cheapest = safe[cheapestPos]
        if (worth(best) >= this.cfg.swapMargin * worth(cheapest) && worth(cheapest) !== Infinity) {
            const safeIndex = Math.trunc(Number(cheapest.index || cheapest.reference || cheapestPos + 1))
            const params = { index: best.index, safe_index: safeIndex }
            if (this.allowed(floor, 'swap_box', params)) {
                return decision(
                    'swap_box',
                    params,
                    `用 ${best.name}（${int(worth(best))}）替换 ${cheapest.name}（${int(worth(cheapest))}）`
                )
            }
        }
        return null
    }

    protected fightDecision(state: Json, ctr: Map<string, Json>): Decision {
        const run = state.run
        const foe = run.event.enemy
        const boss = Boolean(foe.boss)
        const est = this.estimate(state)
        const limit = boss ? this.cfg.maxBossWindowRisk : this.cfg.maxDeathRisk
        const tag = `${boss ? '首领' : foe.elite ? '精英' : '小怪'} ${foe.name}`
        const topup = this.combatOnlyTopup(run)
        if (topup) return { ...topup, reason: `${tag}：${topup.reason}` }
        if (est.death <= limit)
            return decision('fight', {}, `${tag}：死亡 ${pct(est.death, 1)} 胜率 ${pct(est.win)}`, est)
        // 每步只花一项资源；循环会在每次之后重新评估
        const options = [
            () => this.bestHeal(run, true, false),
            () => this.combatBuff(run, boss),
            () => this.bestHeal(run, true, true)
        ]
        for (const option of options) {
            const d = option()
            if (d) return { ...d, reason: `${tag}：死亡 ${pct(est.death, 1)} → ${d.reason}`, estimate: est }
        }
        const grenade = BossFarmPolicy.bagItem(run, 'grenade')
        if (grenade && est.death > limit) {
            return decision('use', { index: grenade.index }, `${tag}：手雷（死亡 ${pct(est.death, 1)}）`, est)
        }
        const dry = this.cfg.extractWhenDry && !this.hasHealing(run)
        if (!this.greedy(run) && (est.death > this.cfg.abortRisk || (dry && est.death > limit))) {
            if (ctr.has('extract'))
                return decision('extract', {}, `${tag}：死亡 ${pct(est.death, 1)}，资源耗尽 → 撤离`, est)
            const smoke = BossFarmPolicy.bagItem(run, 'smoke')
            if (smoke && !boss)
                return decision('use', { index: smoke.index }, `${tag}：烟雾弹脱战（死亡 ${pct(est.death, 1)}）`, est)
            if (ctr.has('flee')) {
                return decision('flee', {}, `${tag}：逃跑（${ctr.get('flee').label}），死亡 ${pct(est.death, 1)}`, est)
            }
        }
        return decision('fight', {}, `${tag}：被迫迎战，死亡 ${pct(est.death, 1)}`, est)
    }

    private choice(state: Json): Decision {
        const run = state.run
        const hpFrac = run.hp / run.max_hp
        let best: Json = null
        let bestScore = -Infinity
        for (const c of (state.controls || []) as Json[]) {
            if (c.command !== 'event' || !this.allowed(run.floor, 'event', c.params)) continue
            const score = BossFarmPolicy.scoreOption(c.description || '', hpFrac, run.max_hp, this.eventAtkCapped)
            if (score > bestScore) {
                best = c
                bestScore = score
            }
        }
        if (best === null || bestScore < 0) return decision('leave', {}, `奇遇 ${run.event.name}：没有值得的选项`)
        return decision('event', best.params, `奇遇 ${run.event.name}：${best.label}（${fixed(bestScore, 0)}）`)
    }

    /** 以 HP 当量打分；未知收益按小幅正值计 */
    static scoreOption(desc: string, hpFrac: number, maxHp: number, atkCapped = false): number {
        if (desc.includes('无消耗')) return 0.0
        const m = /(\d+)%(?:成功|→)/.exec(desc)
        const p = m ? Number(m[1]) / 100 : 1.0
        const parts = desc.split('失败')
        const success = parts[0]
        const failure = parts.length > 1 ? parts[1] : ''
        let gain = 5.0 // 战利品 / 杂项
        const heal = pctOf(success, /回(\d+)%血/)
        gain += heal * maxHp * Math.min(1.0, (1 - hpFrac) * 2)
        if (!atkCapped) gain += pctOf(success, /攻\+(\d+)%/) * 800 // 本局攻击加成（奇遇上限 +60%）
        gain += pctOf(success, /防\+(\d+)%/) * 500
        if (success.includes('偷看') || success.includes('侦察')) gain += 2
        const m2 = /扣约(\d+)HP/.exec(failure)
        let failCost = m2 ? Number(m2[1]) : 0.0
        let cost = pctOf(desc, /耗 (\d+)% 当前血/) * hpFrac * maxHp
        const m3 = /耗.*?约(\d+)/.exec(desc)
        if (m3 && !cost) cost = Number(m3[1])
        if (hpFrac < 0.5) {
            cost *= 2
            failCost *= 2
        }
        return p * gain - (1 - p) * failCost * 1.5 - cost
    }

    private scoreWear(desc: string): number {
        if (desc.includes('伤害 ×2')) return this.cfg.balanceGearScore
        return BossFarmPolicy.wearScore(desc)
    }

    private fieldOffer(run: Json): Decision {
        const offer = run.field_offer
        const newScore = this.scoreWear(offer.description || '')
        const slots = (run.field_slots || []) as Json[]
        const choices = ((offer.choices || []) as Json[]).map(c => c.control.params)
        const discard = choices.find(p => p.slot === 0)
        if (!discard) throw new AgentError(`临装 ${offer.name} 没有放弃选项`)
        if (newScore <= 0) return decision('field_pick', discard, `跳过临装 ${offer.name}（评分 ${fixed(newScore)}）`)
        const free = slots.flatMap((s, i) => (isEmpty(s) ? [i + 1] : []))
        if (free.length) {
            const target = choices.find(p => p.slot === free[0])
            if (target) return decision('field_pick', target, `佩戴 ${offer.name}（评分 ${fixed(newScore)}）`)
        }
        let worstSlot: number | null = null
        let worstScore = Infinity
        slots.forEach((s, i) => {
            const desc = s !== null && typeof s === 'object' ? s.description || '' : ''
            const score = desc ? this.scoreWear(desc) : 0.0
            if (score < worstScore) {
                worstSlot = i + 1
                worstScore = score
            }
        })
        const target = choices.find(p => p.slot === worstSlot)
        if (target && newScore > worstScore + 1) {
            return decision(
                'field_pick',
                target,
                `临装位 ${worstSlot}（${fixed(worstScore)}）换成 ${offer.name}（${fixed(newScore)}）`
            )
        }
        return decision('field_pick', discard, `保留当前临装，不换 ${offer.name}`)
    }

    static wearScore(desc: string): number {
        let score = 0.0
        for (const [, name, value] of desc.matchAll(/([一-龥]+)([+-]\d+)/g)) {
            const weight = WEAR_WEIGHTS.find(([key]) => name.endsWith(key))?.[1] ?? 0.3
            score += weight * Number(value)
        }
        return score
    }

    /** 本局捡到的替身娃娃会在死亡时把背包带回家，因此永远不需要撤离 */
    greedy(run: Json): boolean {
        return this.cfg.greedyWithDoll && BossFarmPolicy.bagItem(run, 'doll') !== null
    }

    hasHealing(run: Json): boolean {
        return BossFarmPolicy.fieldHeals(run).length > 0 || BossFarmPolicy.bagHeals(run, true).length > 0
    }

    protected advance(state: Json, ctr: Map<string, Json>): Decision {
        const run = state.run
        if (this.cfg.maxFloor && run.floor >= this.cfg.maxFloor && ctr.has('extract')) {
            return decision('extract', {}, `已到最高层 ${this.cfg.maxFloor}`)
        }
        if (this.cfg.extractWhenDry && ctr.has('extract') && !this.hasHealing(run) && !this.greedy(run)) {
            return decision('extract', {}, '回复手段已用完 → 撤离，带回背包')
        }
        const hpFrac = run.hp / run.max_hp
        const heal = this.bestHeal(run, false, false)
        if (heal && (hpFrac < this.cfg.rushMinHp || heal.command === 'field_use')) return heal
        if (hpFrac < this.cfg.rushMinHp) {
            const forced = this.bestHeal(run, false, true)
            if (forced) return forced
            if (ctr.has('down')) {
                const combatOnly = BossFarmPolicy.bagHeals(run, true).some(([, it]) => COMBAT_ONLY.has(it.id))
                const why = combatOnly ? '只剩战斗专用回复' : '没有回复'
                return decision('down', {}, `HP 低（${pct(hpFrac)}），${why}：单步前进，遇敌时才能回血`)
            }
        }
        if (ctr.has('rush')) {
            const risk = chainRisk(this.lastingBuild(state), state, this.cal, 4, this.cfg.samples, run.map || 'mall')
            if (risk <= this.cfg.maxRushRisk) return decision('rush', {}, `连战（连续风险 ${pct(risk, 1)}）`)
            const forced = this.bestHeal(run, false, true)
            if (forced && hpFrac < 0.95) return { ...forced, reason: `连战风险 ${pct(risk, 1)} → ${forced.reason}` }
            if (ctr.has('down')) return decision('down', {}, `连战风险 ${pct(risk, 1)}：单步前进，每场战斗都能准备`)
            return decision('rush', {}, `连战（风险 ${pct(risk, 1)}，别无选择）`)
        }
        return decision('down', {}, '下一层')
    }
}

/** Python 真值语义下的「空」：None、空 dict / list / 字符串 */
function isEmpty(v: unknown): boolean {
    if (v === null || v === undefined || v === '' || v === 0 || v === false) return true
    if (Array.isArray(v)) return v.length === 0
    if (typeof v === 'object') return Object.keys(v as object).length === 0
    return false
}

/** 对应 Python 的 RuntimeError：可在监督循环里重试的逻辑错误 */
export class AgentError extends Error {
    constructor(message: string) {
        super(message)
        this.name = 'AgentError'
    }
}
