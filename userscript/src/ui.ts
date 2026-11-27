// ============================================================================
// ui.ts — 游戏页面右下角的悬浮面板：设置、开始 / 停止、实时状态、日志、导出
//
// 面板放在 Shadow DOM 中：游戏的样式不会影响面板，面板的样式也不会影响游戏。
// ============================================================================

import { ElevatorClient } from './client.ts'
import { Farmer } from './farmer.ts'
import type { HistoryRow, LogLevel, Progress } from './farmer.ts'
import { int } from './format.ts'
import { TabLock } from './lock.ts'
import { BLESSING_SETS, FOCUS_AUTO } from './heirlooms.ts'
import { DEFAULT_SETTINGS, MAPS, normalizeSettings, parseLoadout, validFocus } from './settings.ts'
import type { Settings } from './settings.ts'
import { load, save } from './storage.ts'

declare const __APP_VERSION__: string

const LOG_LIMIT = 400

const CSS = `
:host { all: initial; }
.panel {
    --bg: #16181d; --fg: #e6e6e6; --muted: #9aa0a6; --line: #2c3038; --accent: #4f9dff;
    --good: #5ccf7a; --warn: #f0b545; --error: #ff6b6b; --field: #1f2229;
    position: fixed; right: 12px; bottom: 12px; z-index: 2147483000;
    width: min(380px, calc(100vw - 24px)); max-height: calc(100vh - 24px);
    display: flex; flex-direction: column; box-sizing: border-box;
    background: var(--bg); color: var(--fg); border: 1px solid var(--line); border-radius: 10px;
    box-shadow: 0 8px 28px rgba(0,0,0,.45);
    font: 12px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
@media (prefers-color-scheme: light) {
    .panel { --bg: #ffffff; --fg: #1d1f23; --muted: #5f6368; --line: #dadce0; --field: #f5f6f8;
             --good: #188038; --warn: #b06000; --error: #c5221f; --accent: #1a73e8; }
}
.panel * { box-sizing: border-box; }
header { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid var(--line); cursor: default; }
header .title { font-weight: 600; flex: 1; }
header .ver { color: var(--muted); font-weight: 400; margin-left: 4px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.dot.on { background: var(--good); }
button { font: inherit; color: var(--fg); background: var(--field); border: 1px solid var(--line);
         border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover:not(:disabled) { border-color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.icon { padding: 0 8px; line-height: 20px; }
.body { display: flex; flex-direction: column; gap: 8px; padding: 10px; overflow: auto; min-height: 0; }
.panel.collapsed .body { display: none; }
.live { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px 8px; }
.live div { display: flex; flex-direction: column; }
.live span { color: var(--muted); font-size: 11px; }
.live b { font-variant-numeric: tabular-nums; font-weight: 600; }
.row { display: flex; flex-wrap: wrap; gap: 6px; }
details { border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; }
summary { cursor: pointer; color: var(--muted); }
.form { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin-top: 8px; }
.form label { display: flex; flex-direction: column; gap: 2px; color: var(--muted); }
.form label.wide { grid-column: 1 / -1; }
.form label.check { flex-direction: row; align-items: center; gap: 6px; color: var(--fg); grid-column: 1 / -1; }
input, select { font: inherit; color: var(--fg); background: var(--field); border: 1px solid var(--line);
                border-radius: 6px; padding: 3px 6px; width: 100%; min-width: 0; }
input[type=checkbox] { width: auto; }
.log { background: var(--field); border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px;
       height: 220px; overflow: auto; font: 11px/1.45 ui-monospace, Consolas, "Cascadia Mono", monospace;
       white-space: pre-wrap; word-break: break-all; }
.log .t { color: var(--muted); }
.log .good { color: var(--good); } .log .warn { color: var(--warn); } .log .error { color: var(--error); }
.hint { color: var(--muted); font-size: 11px; }
`

function h<K extends keyof HTMLElementTagNameMap>(
    tag: K,
    props: Partial<HTMLElementTagNameMap[K]> & Record<string, unknown> = {},
    ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
    const el = document.createElement(tag)
    Object.assign(el, props)
    el.append(...children)
    return el
}

function download(root: ShadowRoot, name: string, text: string, type: string): void {
    const url = URL.createObjectURL(new Blob([text], { type }))
    const a = h('a', { href: url, download: name })
    root.append(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

function csvCell(v: unknown): string {
    const s = String(v ?? '')
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function historyCsv(rows: HistoryRow[]): string {
    const head = [
        'time',
        'mode',
        'policy',
        'map',
        'floor',
        'bosses',
        'rainbows',
        'steps',
        'seconds',
        'outcome',
        'summary'
    ]
    const lines = rows.map(r =>
        [
            r.time,
            r.mode,
            r.policy ?? 'threshold',
            r.map,
            r.floor,
            r.bosses,
            r.rainbows.join('|'),
            r.steps,
            r.seconds,
            r.outcome,
            r.summary
        ]
            .map(csvCell)
            .join(',')
    )
    // BOM：让 Excel 按 UTF-8 打开中文
    return `﻿${[head.join(','), ...lines].join('\n')}\n`
}

function confirmReal(s: Settings): boolean {
    const costs = ['消耗门票']
    if (s.buyTickets) costs.push('门票用完后用金币购买付费门票')
    if (s.restock) costs.push(`用金币补货（保留 ${int(s.goldReserve)}）`)
    if (s.sellLoot) costs.push('出售本局战利品')
    return confirm(`正式模式将：${costs.join('、')}。\n\n确定开始最多 ${s.runs} 局吗？`)
}

export function mountPanel(): void {
    const host = h('div', { id: 'elevator-farmer-host' })
    document.body.append(host)
    const root = host.attachShadow({ mode: 'open' })
    root.append(h('style', { textContent: CSS }))

    let settings: Settings = normalizeSettings(load('settings', DEFAULT_SETTINGS))
    const client = new ElevatorClient()
    const lock = new TabLock()
    let farmer: Farmer | null = null
    let running = false

    // ---------------------------------------------------------------- 结构
    const dot = h('span', { className: 'dot' })
    const collapseBtn = h('button', { className: 'icon', title: '收起 / 展开' }, '—')
    const header = h(
        'header',
        {},
        dot,
        h('span', { className: 'title' }, '无尽电梯 · 刷首领', h('span', { className: 'ver' }, `v${__APP_VERSION__}`)),
        collapseBtn
    )

    const liveCells: Record<string, HTMLElement> = {}
    const live = h('div', { className: 'live' })
    for (const [key, label] of [
        ['state', '状态'],
        ['run', '局数'],
        ['floor', '楼层'],
        ['hp', 'HP'],
        ['bosses', '本局首领'],
        ['total', '累计首领'],
        ['rainbows', '彩虹'],
        ['mode', '模式']
    ]) {
        liveCells[key] = h('b', {}, '–')
        live.append(h('div', {}, h('span', {}, label), liveCells[key]))
    }

    const startBtn = h('button', { className: 'primary' }, '开始')
    const stopAfterBtn = h('button', { disabled: true, title: '打完当前这一局后停止' }, '本局后停止')
    const stopNowBtn = h('button', { disabled: true, title: '当前请求返回后立即停下，对局保留在服务器上' }, '立即暂停')
    const statusBtn = h('button', { title: '读取图鉴、金币与门票' }, '查看状态')
    const dedupeBtn = h('button', { title: '按「每种只留一件」规则出售仓库里重复的红 / 金物品（不开局）' }, '仓库去重')
    const buttons = h('div', { className: 'row' }, startBtn, stopAfterBtn, stopNowBtn, statusBtn, dedupeBtn)

    // ---------------------------------------------------------------- 设置表单
    const form = h('div', { className: 'form' })
    const inputs: HTMLElement[] = []
    /** 运行状态变化后重新计算的启用状态（例如只在「指定两套」时可用的下拉框） */
    const refreshers: (() => void)[] = []
    const persist = () => save('settings', settings)

    function field(label: string, input: HTMLInputElement | HTMLSelectElement, cls = ''): void {
        inputs.push(input)
        const wrap = h('label', { className: cls })
        if (cls === 'check') wrap.append(input, label)
        else wrap.append(label, input)
        form.append(wrap)
    }

    function select<K extends 'mode' | 'policy' | 'map'>(key: K, label: string, options: [string, string][]): void {
        const el = h('select')
        for (const [value, caption] of options) el.append(h('option', { value, textContent: caption }))
        el.value = settings[key]
        el.addEventListener('change', () => {
            settings = { ...settings, [key]: el.value }
            persist()
        })
        field(label, el, key === 'map' ? 'wide' : '')
    }

    /** 追寻祝福：沿用游戏内选择 / 自动（缺件最多的两套）/ 指定两套（两个下拉框，只在「指定」时生效） */
    function focusPicker(): void {
        const mode = h('select')
        for (const [value, caption] of [
            ['last', '沿用游戏内上次的选择'],
            [FOCUS_AUTO, '自动：缺件最多的两套'],
            ['manual', '指定两套']
        ]) {
            mode.append(h('option', { value, textContent: caption }))
        }
        const pickers = [h('select'), h('select')]
        const [first, second] = settings.focus.includes(',') ? settings.focus.split(',') : ['B06', 'B08']
        pickers.forEach((el, i) => {
            for (const [id, name] of BLESSING_SETS) {
                el.append(h('option', { value: id, textContent: `${id.slice(1).replace(/^0/, '')}. ${name}` }))
            }
            el.value = i === 0 ? first : second
        })
        mode.value = settings.focus === '' ? 'last' : settings.focus === FOCUS_AUTO ? FOCUS_AUTO : 'manual'
        const sync = () => {
            const wanted =
                mode.value === 'last'
                    ? ''
                    : mode.value === FOCUS_AUTO
                      ? FOCUS_AUTO
                      : `${pickers[0].value},${pickers[1].value}`
            refresh()
            if (!validFocus(wanted)) return // 两套相同：不保存，保留原设置
            settings = { ...settings, focus: wanted }
            persist()
        }
        mode.addEventListener('change', sync)
        pickers.forEach(el => el.addEventListener('change', sync))
        field('追寻祝福（正式下潜）', mode, 'wide')
        field('指定：第一套', pickers[0])
        field('指定：第二套', pickers[1])
        const refresh = () => pickers.forEach(el => (el.disabled = running || mode.value !== 'manual'))
        refreshers.push(refresh)
        refresh()
    }

    type NumKey = 'runs' | 'goldReserve' | 'delay' | 'balanceGearScore'
    function number(
        key: NumKey | 'maxFloor',
        label: string,
        opts: { min?: number; step?: number; placeholder?: string }
    ): void {
        const el = h('input', { type: 'number', step: String(opts.step ?? 1), placeholder: opts.placeholder ?? '' })
        if (opts.min !== undefined) el.min = String(opts.min)
        el.value = settings[key] === null ? '' : String(settings[key])
        el.addEventListener('change', () => {
            const v = el.value.trim() === '' ? null : Number(el.value)
            if (key === 'maxFloor') {
                settings = { ...settings, maxFloor: v !== null && Number.isInteger(v) && v > 0 ? v : null }
            } else if (v !== null && Number.isFinite(v) && (opts.min === undefined || v >= opts.min)) {
                settings = { ...settings, [key]: key === 'runs' ? Math.floor(v) : v }
            }
            el.value = settings[key] === null ? '' : String(settings[key])
            persist()
        })
        field(label, el)
    }

    type TextKey = 'loadout' | 'sellRarities' | 'dedupeRarities'
    function text(key: TextKey, label: string): void {
        const el = h('input', { type: 'text', value: settings[key], spellcheck: false })
        el.addEventListener('change', () => {
            settings = { ...settings, [key]: el.value.trim() }
            persist()
        })
        field(label, el, 'wide')
    }

    type BoolKey = 'restock' | 'rushElites' | 'buyTickets' | 'sellLoot' | 'verbose'
    function check(key: BoolKey, label: string): void {
        const el = h('input', { type: 'checkbox', checked: settings[key] })
        el.addEventListener('change', () => {
            settings = { ...settings, [key]: el.checked }
            persist()
        })
        field(label, el, 'check')
    }

    select('mode', '模式', [
        ['sim', '练习（免费，无收益）'],
        ['real', '正式（消耗门票与金币）']
    ])
    select('policy', '决策', [
        ['threshold', '阈值规则'],
        ['markov', '马尔可夫（后台求解约 5 分钟、内存约 600 MB；求解完成前用阈值规则）']
    ])
    number('runs', '局数', { min: 1 })
    select('map', '地图', MAPS)
    focusPicker()
    number('maxFloor', '最高层（到达即撤离）', { min: 1, placeholder: '不限' })
    number('goldReserve', '金币保留线', { min: 0, step: 10000 })
    text('loadout', '携带物资（物品id=数量，逗号分隔）')
    check('restock', '开局前自动补货（不低于金币保留线）')
    check('rushElites', '连战自动打精英（模型判断同样安全时；每局停顿更少）')
    check('buyTickets', '免费门票用完后购买付费门票（350 金币）')
    check('sellLoot', '正式局结束后出售本局战利品')
    text('sellRarities', '整批出售的稀有度')
    text('dedupeRarities', '每种只留一件（价值最高）的稀有度')
    number('balanceGearScore', '「伤害 ×2」临装评分（负数 = 不戴）', { step: 0.5 })
    number('delay', '请求最短间隔（秒）', { min: 0, step: 0.01 })
    check('verbose', '详细日志（记录每一步）')

    const settingsBox = h('details', {}, h('summary', {}, '设置'), form)

    // ---------------------------------------------------------------- 日志与导出
    const logBox = h('div', { className: 'log' })
    const csvBtn = h('button', { title: '每局一行：楼层、首领数、彩虹、耗时……' }, '导出记录 CSV')
    const traceBtn = h('button', { title: '最近 500 步的决策与服务器结果' }, '导出轨迹')
    const clearBtn = h('button', {}, '清空日志')
    const footer = h('div', { className: 'row' }, csvBtn, traceBtn, clearBtn)
    const hint = h(
        'div',
        { className: 'hint' },
        '运行期间请保持本标签页打开，不要在游戏界面中操作；结束后刷新页面即可看到最新状态。'
    )

    const panel = h(
        'div',
        { className: 'panel' },
        header,
        h('div', { className: 'body' }, live, buttons, settingsBox, logBox, footer, hint)
    )
    root.append(panel)

    // ---------------------------------------------------------------- 行为
    const ui = load<{ collapsed?: boolean }>('ui', {})
    const setCollapsed = (c: boolean) => {
        panel.classList.toggle('collapsed', c)
        collapseBtn.textContent = c ? '▢' : '—'
        save('ui', { ...ui, collapsed: c })
    }
    setCollapsed(Boolean(ui.collapsed))
    collapseBtn.addEventListener('click', () => setCollapsed(!panel.classList.contains('collapsed')))

    function log(message: string, level: LogLevel = 'info'): void {
        const stamp = new Date().toLocaleTimeString('zh-CN', { hour12: false })
        const atBottom = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 4
        logBox.append(h('div', { className: level }, h('span', { className: 't' }, `[${stamp}] `), message))
        while (logBox.childElementCount > LOG_LIMIT) logBox.firstElementChild?.remove()
        if (atBottom) logBox.scrollTop = logBox.scrollHeight
    }

    function showProgress(p: Progress): void {
        liveCells.state.textContent = p.running ? '运行中' : '空闲'
        liveCells.run.textContent = p.run ? `${p.run}/${p.runs}` : '–'
        liveCells.floor.textContent = p.floor ? `F${p.floor}` : '–'
        liveCells.hp.textContent = p.maxHp ? `${int(p.hp)}/${int(p.maxHp)}` : '–'
        liveCells.bosses.textContent = String(p.bosses)
        liveCells.total.textContent = String(p.totalBosses)
        liveCells.rainbows.textContent = String(p.rainbows.length)
        liveCells.mode.textContent = p.mode === 'real' ? '正式' : '练习'
    }

    function setRunning(on: boolean): void {
        running = on
        dot.classList.toggle('on', on)
        startBtn.disabled = on
        statusBtn.disabled = on
        dedupeBtn.disabled = on
        stopAfterBtn.disabled = !on
        stopNowBtn.disabled = !on
        for (const el of inputs) (el as HTMLInputElement).disabled = on
        for (const fn of refreshers) fn()
        if (!on) liveCells.state.textContent = '空闲'
    }

    async function launch(action: 'farm' | 'status' | 'dedupe'): Promise<void> {
        const s = settings
        const statusOnly = action === 'status'
        if (action === 'farm') {
            try {
                parseLoadout(s.loadout)
            } catch (err) {
                log((err as Error).message, 'error')
                return
            }
            if (s.mode === 'real' && !confirmReal(s)) return
        }
        if (action === 'dedupe') {
            const rarities = s.dedupeRarities.trim()
            if (!rarities) return log('「每种只留一件的稀有度」为空，没有要去重的物品', 'warn')
            const ok = confirm(
                `将出售仓库中稀有度为 ${rarities} 的重复物品：每种只保留价值最高的一件，其余全部出售。\n\n确定吗？`
            )
            if (!ok) return
        }
        if (!statusOnly) {
            if (!lock.acquire()) {
                log('另一个标签页正在运行脚本：请先在那边停止，或关闭那个标签页后等待 3 分钟。', 'error')
                return
            }
        }
        const current = new Farmer(s, { log, progress: showProgress }, client, () => lock.beat())
        farmer = current
        setRunning(true)
        const beat = statusOnly ? undefined : setInterval(() => lock.beat(), 20_000)
        try {
            await current.run(action)
        } catch (err) {
            console.error('[elevator-farmer]', err)
        } finally {
            if (beat) clearInterval(beat)
            if (!statusOnly) lock.release()
            setRunning(false)
        }
    }

    startBtn.addEventListener('click', () => void launch('farm'))
    statusBtn.addEventListener('click', () => void launch('status'))
    dedupeBtn.addEventListener('click', () => void launch('dedupe'))
    stopAfterBtn.addEventListener('click', () => {
        farmer?.requestStopAfterRun()
        stopAfterBtn.disabled = true
    })
    stopNowBtn.addEventListener('click', () => {
        farmer?.requestStopNow()
        stopNowBtn.disabled = true
        stopAfterBtn.disabled = true
    })
    csvBtn.addEventListener('click', () => {
        const rows = load<HistoryRow[]>('history', [])
        if (!rows.length) return log('还没有已完成的对局记录', 'warn')
        download(root, 'elevator-runs.csv', historyCsv(rows), 'text/csv;charset=utf-8')
    })
    traceBtn.addEventListener('click', () => {
        if (!farmer?.trace.length) return log('本次会话还没有轨迹', 'warn')
        const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')
        download(root, `elevator-trace-${stamp}.jsonl`, `${farmer.trace.join('\n')}\n`, 'application/x-ndjson')
    })
    clearBtn.addEventListener('click', () => logBox.replaceChildren())

    window.addEventListener('beforeunload', e => {
        if (running) e.preventDefault()
    })

    showProgress({
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
    })
    log(`已加载 v${__APP_VERSION__}。默认是练习模式；在「设置」中可切换正式模式与各项参数。`)
}
