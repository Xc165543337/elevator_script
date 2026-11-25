// ============================================================================
// farmer.ts — 刷首领运行器（移植自 farm.py）
//
// 门票、补货、开局 / 续局、记录、彩虹检测；所有彩虹解锁后自动停止。
// 与 Python 版不同：不登录（沿用页面会话），记录保存在浏览器中，可随时暂停。
// ============================================================================

import { Calibration } from './combat.ts'
import type { CalibrationData } from './combat.ts'
import { ElevatorClient, GameError, NetworkError } from './client.ts'
import { int } from './format.ts'
import { AgentError, BossFarmPolicy } from './policy.ts'
import type { Decision } from './policy.ts'
import { parseList, parseLoadout } from './settings.ts'
import type { Settings } from './settings.ts'
import { load, save } from './storage.ts'
import type { Json } from './types.ts'

const TICKET_PRICE = 350
/** 只出售这两类；补给（药品、局内道具）、箱子与装备永不出售 */
const SELL_KINDS = new Set(['collection', 'material'])
const TRACE_LIMIT = 500
const HISTORY_LIMIT = 2000

export type LogLevel = 'info' | 'good' | 'warn' | 'error'

export interface Progress {
    running: boolean
    mode: string
    run: number
    runs: number
    floor: number
    hp: number
    maxHp: number
    bosses: number
    totalBosses: number
    rainbows: string[]
}

export interface RunSummary {
    floor: number
    bosses: number
    rainbows: string[]
    steps: number
    seconds: number
    outcome: string
    summary: string
}

export interface HistoryRow extends RunSummary {
    time: string
    mode: string
    map: string
}

export interface FarmerHooks {
    log(message: string, level?: LogLevel): void
    progress(p: Progress): void
}

/** 用户点击「立即暂停」：中断当前等待 / 步骤 */
export class StopError extends Error {
    constructor() {
        super('已暂停')
        this.name = 'StopError'
    }
}

/**
 * 要出售的 (条目, 数量)。
 * dedupeRarities（红 / 金）：整个仓库每种物品只保留价值最高的一件，其余全部出售（含旧的重复品）。
 * bulkRarities（紫 / 绿 / 白）：出售 baseline 之后新增的全部条目（本局战利品）。
 */
export function selectLootToSell(
    stock: Json[],
    baseline: Set<number>,
    dedupeRarities: Set<string>,
    bulkRarities: Set<string>
): [Json, number][] {
    const picked: [Json, number][] = []
    const sellable = stock.filter(it => SELL_KINDS.has(it.kind) && it.stock_id !== null && it.stock_id !== undefined)
    const groups = new Map<string, Json[]>()
    for (const it of sellable) {
        if (dedupeRarities.has(it.rarity)) {
            const list = groups.get(it.id) ?? []
            list.push(it)
            groups.set(it.id, list)
        } else if (bulkRarities.has(it.rarity) && !baseline.has(it.stock_id)) {
            picked.push([it, it.quantity || 1])
        }
    }
    for (const entries of groups.values()) {
        entries.sort((a, b) => (b.value || 0) - (a.value || 0) || b.stock_id - a.stock_id)
        const [keep, ...others] = entries
        if ((keep.quantity || 1) > 1) picked.push([keep, keep.quantity - 1])
        for (const it of others) picked.push([it, it.quantity || 1])
    }
    return picked
}

export class Farmer {
    readonly client: ElevatorClient
    readonly cal: Calibration
    readonly policy: BossFarmPolicy
    readonly trace: string[] = []
    private stopNow = false
    private stopAfterRun = false
    private wakeSleepers: (() => void)[] = []
    private lastCallEnd = 0
    private tally: {
        t0: number
        steps: number
        bosses: number
        rainbows: string[]
        floor: number
        outcome: string
        summary: string
    }
    private progress: Progress

    constructor(
        readonly settings: Settings,
        private readonly hooks: FarmerHooks,
        client?: ElevatorClient,
        private readonly heartbeat: () => void = () => {}
    ) {
        this.client = client ?? new ElevatorClient()
        this.cal = new Calibration(load<Partial<CalibrationData>>('calibration', {}))
        this.policy = new BossFarmPolicy(this.cal, {
            maxFloor: settings.maxFloor,
            balanceGearScore: settings.balanceGearScore
        })
        this.tally = this.freshTally()
        this.progress = {
            running: false,
            mode: settings.mode,
            run: 0,
            runs: settings.runs,
            floor: 0,
            hp: 0,
            maxHp: 0,
            bosses: 0,
            totalBosses: 0,
            rainbows: []
        }
    }

    // ------------------------------------------------------------- 控制
    /** 本局结束后停止 */
    requestStopAfterRun(): void {
        this.stopAfterRun = true
        this.log('将在本局结束后停止')
    }

    /** 立即暂停：当前请求返回后停下，本局保留在服务器上 */
    requestStopNow(): void {
        this.stopNow = true
        for (const wake of this.wakeSleepers.splice(0)) wake()
    }

    private checkStop(): void {
        if (this.stopNow) throw new StopError()
    }

    /** 可被「立即暂停」打断的等待 */
    private sleep(ms: number): Promise<void> {
        this.checkStop()
        return new Promise((resolve, reject) => {
            const wake = () => {
                clearTimeout(timer)
                reject(new StopError())
            }
            const timer = setTimeout(() => {
                this.wakeSleepers = this.wakeSleepers.filter(w => w !== wake)
                resolve()
            }, ms)
            this.wakeSleepers.push(wake)
        })
    }

    // ------------------------------------------------------------- 工具
    private log(message: string, level: LogLevel = 'info'): void {
        this.hooks.log(message, level)
    }

    private emit(patch: Partial<Progress>): void {
        this.progress = { ...this.progress, ...patch }
        this.hooks.progress(this.progress)
    }

    /**
     * 带节流的命令发送，处理过期 revision 与网络错误。
     * 节流只补足距上次请求结束的剩余间隔：网络延迟本身已超过间隔时不再设定时器，
     * 这样后台标签页（定时器被限速到 ≥ 1 秒）也不会明显变慢。
     */
    async call(command: string, params: Record<string, unknown> = {}): Promise<Json> {
        let attempt = 0
        let cooldowns = 0
        while (attempt < 5) {
            this.checkStop()
            const wait = this.settings.delay * 1000 - (Date.now() - this.lastCallEnd)
            if (wait > 0) await this.sleep(wait)
            try {
                return await this.client.command(command, params)
            } catch (err) {
                if (err instanceof GameError) {
                    if (err.code === 'cooldown' && cooldowns < 30) {
                        // 服务器限速（「操作太快」）
                        cooldowns += 1
                        await this.sleep(1000)
                        continue
                    }
                    attempt += 1
                    if (err.code === 'maintenance') {
                        this.log('服务器维护中，60 秒后重试', 'warn')
                        await this.sleep(60_000)
                    } else if (
                        attempt < 3 &&
                        !(err.code === null || err.code === 'invalid' || err.code === 'forbidden')
                    ) {
                        this.log(`重试：${err.message}`, 'warn')
                    } else {
                        throw err
                    }
                    await this.client.refresh()
                } else if (err instanceof NetworkError) {
                    attempt += 1
                    this.log(`网络错误（${err.message}），稍后重试`, 'warn')
                    await this.sleep(2 ** attempt * 1000)
                    await this.safeRefresh()
                } else {
                    throw err
                }
            } finally {
                this.lastCallEnd = Date.now()
                this.heartbeat()
            }
        }
        throw new AgentError(`命令 ${command} 多次失败`)
    }

    async safeRefresh(): Promise<boolean> {
        try {
            await this.client.refresh()
            return true
        } catch (err) {
            if (err instanceof GameError || err instanceof NetworkError) {
                this.log(`刷新状态失败（${err.message}）`, 'warn')
                return false
            }
            throw err
        }
    }

    /** playRun，遇到临时的服务器 / 网络故障时继续同一局 */
    private async playRunSupervised(retries = 10): Promise<RunSummary> {
        for (let attempt = 0; ; attempt++) {
            try {
                return await this.playRun()
            } catch (err) {
                const retryable = err instanceof GameError || err instanceof NetworkError || err instanceof AgentError
                if (!retryable || attempt === retries) throw err
                this.log(`本局中断（${(err as Error).message}），30 秒后继续`, 'warn')
                await this.sleep(30_000)
                while (!(await this.safeRefresh())) await this.sleep(30_000)
            }
        }
    }

    rainbowStatus(): [string, boolean][] {
        return ((this.client.state?.atlas?.rainbow || []) as Json[]).map(r => [r.name, Boolean(r.unlocked)])
    }

    printStatus(): void {
        const st = this.client.state
        const atlas = st.atlas || {}
        const player = st.player || {}
        this.log(
            `图鉴 ${atlas.known}/${atlas.total} ｜ 金币 ${int(player.gold || 0)} ｜ 门票 ${player.tickets}（付费 +${player.paid_tickets}）`
        )
        for (const [name, unlocked] of this.rainbowStatus()) {
            this.log(`  彩虹 ${name}：${unlocked ? '已解锁' : '未解锁'}`, unlocked ? 'good' : 'info')
        }
        const missing = ((atlas.items || []) as Json[])
            .filter(i => !i.unlocked && i.rarity !== 'rainbow')
            .map(i => i.name)
        if (missing.length) this.log(`  其他未收集：${missing.join('、')}`)
    }

    // ------------------------------------------------------------- 准备
    private ownedSupplies(): Map<string, number> {
        const stock = new Map<string, number>()
        for (const item of (this.client.state.supplies || []) as Json[]) {
            stock.set(item.id, (stock.get(item.id) ?? 0) + (item.quantity ?? 1))
        }
        return stock
    }

    private async ensureTicket(): Promise<boolean> {
        let player = this.client.state.player
        if (player.tickets + player.paid_tickets > 0) return true
        if (!this.settings.buyTickets || player.gold - TICKET_PRICE < this.settings.goldReserve) return false
        try {
            await this.call('buy', { id: 'ticket' })
        } catch (err) {
            if (!(err instanceof GameError)) throw err
            this.log(`无法购买门票：${err.message}`, 'warn')
            return false
        }
        player = this.client.state.player
        return player.tickets + player.paid_tickets > 0
    }

    private async restock(loadout: Record<string, number>): Promise<void> {
        const prices = new Map<string, number>(((this.client.state.shop || []) as Json[]).map(it => [it.id, it.price]))
        const owned = this.ownedSupplies()
        for (const [itemId, want] of Object.entries(loadout)) {
            const missing = want - (owned.get(itemId) ?? 0)
            if (missing <= 0) continue
            const price = prices.get(itemId)
            if (price === undefined) {
                await this.craft(itemId, missing)
                continue
            }
            const affordable = Math.floor((this.client.state.player.gold - this.settings.goldReserve) / price)
            const qty = Math.min(missing, Math.max(0, affordable))
            if (qty <= 0) {
                this.log('已到金币保留线，停止补货', 'warn')
                return
            }
            try {
                await this.call('buy', { id: itemId, quantity: qty }) // 一次买齐
            } catch (err) {
                if (!(err instanceof GameError)) throw err
                this.log(`批量购买被拒绝（${err.message}），改为逐个购买`, 'warn')
                for (let i = 0; i < qty; i++) await this.call('buy', { id: itemId })
            }
            this.log(`购买 ${qty} × ${itemId}`)
        }
    }

    /** 黑市物品（如扩容袋、手雷）：材料足够时制作 */
    private async craft(itemId: string, count: number): Promise<void> {
        let made = 0
        for (let i = 0; i < count; i++) {
            const recipe = ((this.client.state.market || []) as Json[]).find(m => m.id === itemId)
            if (!recipe || !recipe.ready) break
            await this.call('craft', { id: itemId })
            made += 1
        }
        if (made) this.log(`制作 ${made} × ${itemId}`)
        if (made < count) this.log(`${itemId} 还缺 ${count - made} 个（商店没有，材料不足）`, 'warn')
    }

    private warehouseIds(): Set<number> {
        return new Set(
            ((this.client.state.warehouse || []) as Json[])
                .filter(it => it.stock_id !== null && it.stock_id !== undefined)
                .map(it => it.stock_id)
        )
    }

    private async sellRunLoot(baseline: Set<number>): Promise<void> {
        await this.safeRefresh()
        let earned = 0
        let sold = 0
        const stock = (this.client.state.warehouse || []) as Json[]
        const picks = selectLootToSell(
            stock,
            baseline,
            parseList(this.settings.dedupeRarities),
            parseList(this.settings.sellRarities)
        )
        for (const [it, qty] of picks) {
            try {
                await this.call('sell', { stock_id: it.stock_id, quantity: qty })
            } catch (err) {
                if (!(err instanceof GameError)) throw err
                this.log(`无法出售 ${it.name}（${err.message}）`, 'warn')
                continue
            }
            sold += qty
            earned += (it.value || 0) * qty
        }
        this.log(`出售 ${sold} 件本局物品（约 ${int(earned)} 金币）`)
    }

    private async ensureRushSettings(): Promise<void> {
        const player = this.client.state.player
        if (player.rush_floors !== 10 || !player.rush_skip_valuable) {
            await this.call('rush_settings', { floors: 10, skip_valuable: true })
        }
    }

    // ------------------------------------------------------------- 单局
    private async startRun(loadout: Record<string, number>): Promise<void> {
        const mapId = this.settings.map
        if (this.settings.mode === 'sim') {
            await this.call('simulate', { map_id: mapId })
            return
        }
        const owned = this.ownedSupplies()
        let supplies = Object.entries(loadout)
            .filter(([id]) => (owned.get(id) ?? 0) > 0)
            .map(([id, q]) => ({ id, quantity: Math.min(q, owned.get(id) ?? 0) }))
        try {
            await this.call('start', { map_id: mapId, supplies })
        } catch (err) {
            if (!(err instanceof GameError)) throw err
            // 携带上限未公开：数量逐次减半直到被接受
            this.log(`携带物资被拒绝（${err.message}），减少数量后重试`, 'warn')
            while (supplies.length) {
                supplies = supplies
                    .filter(s => s.quantity > 1)
                    .map(s => ({ id: s.id, quantity: Math.floor(s.quantity / 2) }))
                try {
                    await this.call('start', { map_id: mapId, supplies })
                    return
                } catch (retryErr) {
                    if (!(retryErr instanceof GameError)) throw retryErr
                }
            }
            await this.call('start', { map_id: mapId })
        }
    }

    private freshTally() {
        return { t0: Date.now(), steps: 0, bosses: 0, rainbows: [] as string[], floor: 0, outcome: '', summary: '' }
    }

    /** 打到本局结束；计数保存在 tally 中，续局后仍然有效 */
    private async playRun(): Promise<RunSummary> {
        const t = this.tally
        while (this.client.state?.run) {
            this.checkStop()
            const state = this.client.state
            const run = state.run
            const floor = (t.floor = run.floor)
            this.emit({ floor, hp: run.hp, maxHp: run.max_hp, bosses: t.bosses })
            const d: Decision = this.policy.decide(state)
            const foe = d.command === 'fight' ? (run.event || {}).enemy : null
            const fighter = foe ? this.policy.currentBuild(state) : null
            let data: Json
            try {
                data = await this.call(d.command, d.params)
            } catch (err) {
                if (!(err instanceof GameError) || err.code !== 'invalid') throw err
                this.log(`F${floor} ${d.command} ${JSON.stringify(d.params)} 被拒绝：${err.message}；重新选择`, 'warn')
                this.policy.ban(floor, d)
                await this.client.refresh()
                continue
            }
            t.steps += 1
            const result = data.result || {}
            this.policy.note(result.limit_notices || [])
            const battle = result.battle
            if (foe && fighter && battle?.log) {
                this.cal.observe(fighter, foe, battle.log)
                if (battle.outcome === 'victory' && foe.boss) {
                    t.bosses += 1
                    this.emit({ bosses: t.bosses, totalBosses: this.progress.totalBosses + 1 })
                    this.log(`首领 #${t.bosses} 击败于 F${floor}：${foe.name}`, 'good')
                }
            }
            for (const item of Farmer.loot(result)) {
                if (item.rarity === 'rainbow' || String(item.id ?? '').startsWith('prism_')) {
                    t.rainbows.push(item.name)
                    this.emit({ rainbows: [...this.progress.rainbows, item.name] })
                    this.log(`★★★ 彩虹掉落：${item.name} ★★★`, 'good')
                }
            }
            this.pushTrace({
                t: Date.now() / 1000,
                floor,
                hp: run.hp,
                stats: run.effective_stats,
                dmg_mult: run.damage_multiplier,
                slots: run.field_slots,
                enemy: (run.event || {}).enemy,
                decision: d.command,
                params: d.params,
                reason: d.reason,
                result
            })
            if (this.settings.verbose || !['rush', 'fight', 'field_pick'].includes(d.command) || foe?.boss) {
                this.log(
                    `F${floor} HP ${run.hp}/${run.max_hp} → ${d.command}：${d.reason}${result.summary ? ` ｜ ${result.summary}` : ''}`
                )
            }
            t.summary = result.summary ?? t.summary
            t.outcome = result.outcome ?? t.outcome
        }
        this.saveCalibration()
        return {
            floor: t.floor,
            bosses: t.bosses,
            rainbows: [...t.rainbows],
            steps: t.steps,
            seconds: Math.round((Date.now() - t.t0) / 1000),
            outcome: t.outcome,
            summary: t.summary
        }
    }

    static loot(result: Json): Json[] {
        const items: Json[] = [...(result.battle?.loot || [])]
        items.push(...(result.pickups?.loot || []))
        for (const fl of result.rush?.floors || []) items.push(...(fl.loot || []))
        return items
    }

    private pushTrace(entry: unknown): void {
        this.trace.push(JSON.stringify(entry))
        if (this.trace.length > TRACE_LIMIT) this.trace.splice(0, this.trace.length - TRACE_LIMIT)
    }

    saveCalibration(): void {
        save('calibration', this.cal.toJSON())
    }

    private record(summary: RunSummary): void {
        const rows = load<HistoryRow[]>('history', [])
        rows.push({ time: new Date().toISOString(), mode: this.settings.mode, map: this.settings.map, ...summary })
        save('history', rows.slice(-HISTORY_LIMIT))
    }

    // ------------------------------------------------------------- 主流程
    async run(statusOnly = false): Promise<void> {
        const s = this.settings
        this.emit({ running: true, mode: s.mode, run: 0, runs: s.runs })
        try {
            await this.client.refresh()
            this.printStatus()
            if (statusOnly) return
            const loadout = parseLoadout(s.loadout)
            let totalBosses = 0
            let baseline = this.warehouseIds()
            for (let n = 1; n <= s.runs; n++) {
                if (this.stopAfterRun) {
                    this.log('已按要求停止')
                    break
                }
                await this.client.refresh()
                if (!this.client.state.run) {
                    if (s.mode === 'real') {
                        const rainbows = this.rainbowStatus()
                        if (rainbows.length && rainbows.every(([, unlocked]) => unlocked)) {
                            this.log('彩虹物品已全部收集，停止', 'good')
                            break
                        }
                        if (!(await this.ensureTicket())) {
                            this.log('没有可用门票，停止', 'warn')
                            break
                        }
                        if (s.restock) await this.restock(loadout)
                    }
                    await this.ensureRushSettings()
                    this.policy.newRun()
                    this.tally = this.freshTally()
                    await this.client.refresh()
                    baseline = this.warehouseIds()
                    await this.startRun(loadout)
                    this.log(`第 ${n} 局开始（${s.mode === 'sim' ? '练习' : '正式'}，${s.map}）`)
                } else {
                    const isSim = Boolean(this.client.state.run.simulation)
                    if (isSim !== (s.mode === 'sim')) {
                        this.log(
                            `已有进行中的${isSim ? '练习' : '正式'}局；当前为${s.mode === 'sim' ? '练习' : '正式'}模式，不会动它。请切换到对应模式完成该局。`,
                            'warn'
                        )
                        break
                    }
                    this.tally = this.freshTally()
                    this.log(`继续 F${this.client.state.run.floor} 的进行中对局`)
                }
                this.emit({ run: n, bosses: 0 })
                const summary = await this.playRunSupervised()
                totalBosses += summary.bosses
                if (s.mode === 'real' && s.sellLoot) await this.sellRunLoot(baseline)
                this.record(summary)
                this.log(
                    `第 ${n} 局结束：F${summary.floor}，首领 ${summary.bosses}，${summary.steps} 步，${summary.seconds} 秒 ｜ ${summary.summary} ｜ 累计首领 ${totalBosses}`,
                    'good'
                )
            }
            await this.client.refresh()
            this.printStatus()
            this.log(`校准系数：k_player=${this.cal.kPlayer.toFixed(3)} k_enemy=${this.cal.kEnemy.toFixed(3)}`)
        } catch (err) {
            this.saveCalibration()
            if (err instanceof StopError) {
                this.log('已暂停。进行中的对局保留在服务器上（30 分钟无操作会过期），再次点「开始」即可继续。', 'warn')
            } else if (err instanceof GameError && err.code === 'sign_in_required') {
                this.log('尚未登录：请先在游戏页面登录，再点「开始」。', 'error')
            } else {
                this.log(`出错停止：${(err as Error).message || err}`, 'error')
                throw err
            }
        } finally {
            this.emit({ running: false })
        }
    }
}
