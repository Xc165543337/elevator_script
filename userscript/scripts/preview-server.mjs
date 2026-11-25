// ============================================================================
// preview-server.mjs — 本地预览：模拟游戏页面与 API，加载构建好的油猴脚本
//
// 用法: npm run build && node scripts/preview-server.mjs [端口，默认 5179]
// 打开 http://localhost:5179/elevator/ ：页面使用与正式站相同的内容安全策略，
// 假 API 提供可完整打完的练习局（F11 首领战、F21 撤离），用于检查面板与运行流程。
// 不连接真实游戏服务器。
// ============================================================================

import { readFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { resolve } from 'node:path'

const port = Number(process.argv[2] || 5179)
const script = resolve(import.meta.dirname, '../dist/elevator-farmer.user.js')
const CSP =
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"

let rev = 0
let loggedIn = true
const state = {
    revision: 'r0',
    player: {
        gold: 1_234_567,
        tickets: 18,
        paid_tickets: 0,
        rush_floors: 10,
        rush_skip_valuable: true,
        loadout: { 武器: 1, 护甲: 2, 挂件: 3, 核心: 4 },
        secondary: { crit_multiplier: 1.9, pierce: 0.26, leech: 0.06 },
        stat_sources: {
            hp: [{ source: '基础', op: 'base', value: 666 }],
            atk: [{ source: '基础', op: 'base', value: 150 }],
            def: [{ source: '基础', op: 'base', value: 70 }],
            agi: [{ source: '基础', op: 'base', value: 77 }],
            luk: [{ source: '基础', op: 'base', value: 21 }]
        }
    },
    equipment: [1, 2, 3, 4].map((id, i) => ({
        equip_id: id,
        name: `装备${id}`,
        slot: ['武器', '护甲', '挂件', '核心'][i],
        rarity: 'gold',
        set_name: null,
        stats: {},
        affixes: i === 0 ? [{ id: 'pierce', name: '破甲' }] : [],
        description: ''
    })),
    atlas: {
        known: 41,
        total: 45,
        items: [],
        rainbow: [
            { name: '虹彩之证', unlocked: false },
            { name: '棱镜残片', unlocked: true }
        ]
    },
    warehouse: [
        { stock_id: 11, id: 'g01', name: '鎏金怀表', kind: 'collection', rarity: 'gold', value: 5000, quantity: 1 },
        { stock_id: 12, id: 'g01', name: '鎏金怀表', kind: 'collection', rarity: 'gold', value: 7000, quantity: 2 },
        { stock_id: 13, id: 'p01', name: '紫色零件', kind: 'material', rarity: 'purple', value: 800, quantity: 4 }
    ],
    run: null,
    controls: []
}

const ctl = names => names.map(c => ({ command: c, params: {}, label: c }))
const boss = { name: '商场守夜人', boss: true, hp: 520, atk: 48, def: 9, agi: 12, rounds_fought: 0 }

function command(name, params) {
    let result = { outcome: 'ongoing', summary: '' }
    const run = state.run
    if (name === 'simulate') {
        state.run = {
            map: 'mall',
            floor: 1,
            hp: 666,
            max_hp: 666,
            shield: 0,
            simulation: true,
            effective_stats: { hp: 666, atk: 150, def: 70, agi: 77, luk: 21 },
            event: { kind: 'ready' },
            bag: [{ id: 'coffee', index: 1, name: '冰美式续命液', healing: true }],
            safe: [],
            safe_capacity: 4,
            field: [],
            field_slots: [null, null]
        }
        result.summary = '练习局开始'
    } else if (name === 'rush' && run) {
        run.floor += 10
        run.hp -= 60
        result.summary = `连战 10 层，到达 F${run.floor}`
        if (run.floor === 11) run.event = { kind: 'combat', enemy: { ...boss } }
    } else if (name === 'fight' && run?.event?.enemy) {
        const log = [
            'R1：你打 179（怪剩341） ｜ 怪打 40（你剩566，盾挡0）',
            'R2：你打 150（怪剩191）',
            'R3：你打 191（怪剩0）'
        ]
        result = {
            outcome: 'ongoing',
            summary: '击败首领',
            battle: { outcome: 'victory', log, loot: [{ id: 'prism_a', name: '虹彩之证', rarity: 'rainbow' }] }
        }
        run.event = { kind: 'ready' }
        run.hp = run.max_hp
    } else if (name === 'extract' && run) {
        state.run = null
        result = { outcome: 'extracted', summary: '撤离成功' }
    } else if (name === 'rush_settings') {
        result.summary = '已设置'
    } else if (name === 'sell') {
        const it = state.warehouse.find(w => w.stock_id === params?.stock_id)
        if (!it) return { status: 400, body: { error: { code: 'invalid', message: '没有该物品' } } }
        it.quantity -= params.quantity
        state.warehouse = state.warehouse.filter(w => w.quantity > 0)
        result.summary = `出售 ${it.name} × ${params.quantity}`
    } else {
        return { status: 400, body: { error: { code: 'invalid', message: `不能执行 ${name}` } } }
    }
    rev += 1
    state.revision = `r${rev}`
    const r = state.run
    state.controls = !r
        ? []
        : r.event.enemy
          ? ctl(['fight', 'flee'])
          : r.floor >= 21
            ? ctl(['extract'])
            : ctl(['rush', 'down', 'extract'])
    return { status: 200, body: { result, state, sync_token: `t${rev}` } }
}

const PAGE = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>无尽电梯（本地预览）</title>
<style>body{margin:0;font-family:system-ui;background:#0e1013;color:#ccc;padding:24px}</style></head>
<body><h1>无尽电梯 · 本地预览</h1><p>模拟的游戏页面（假 API，不连接真实服务器）。</p>
<p><a href="/elevator/logout">模拟退出登录</a> · <a href="/elevator/login">模拟登录</a></p>
<script src="/elevator/farmer.user.js"></script></body></html>`

createServer((req, res) => {
    const send = (status, body, type = 'application/json') => {
        res.writeHead(status, { 'Content-Type': type, 'Content-Security-Policy': CSP, 'Cache-Control': 'no-store' })
        res.end(typeof body === 'string' ? body : JSON.stringify(body))
    }
    const url = req.url ?? '/'
    if (url === '/elevator/farmer.user.js')
        return send(200, readFileSync(script, 'utf-8'), 'text/javascript; charset=utf-8')
    if (url === '/elevator/logout' || url === '/elevator/login') {
        loggedIn = url.endsWith('login')
        res.writeHead(302, { Location: '/elevator/' })
        return res.end()
    }
    if (url.startsWith('/elevator/api/')) {
        if (!loggedIn) return send(401, { error: { code: 'sign_in_required', message: '请登录后继续' } })
        if (url === '/elevator/api/game/state') return setTimeout(() => send(200, { state, sync_token: `t${rev}` }), 80)
        let raw = ''
        req.on('data', c => (raw += c))
        req.on('end', () => {
            const { command: name, params } = JSON.parse(raw || '{}')
            const out = command(name, params)
            setTimeout(() => send(out.status, out.body), 250)
        })
        return
    }
    if (url.startsWith('/elevator')) return send(200, PAGE, 'text/html; charset=utf-8')
    res.writeHead(302, { Location: '/elevator/' })
    res.end()
}).listen(port, () => console.log(`预览：http://localhost:${port}/elevator/`))
