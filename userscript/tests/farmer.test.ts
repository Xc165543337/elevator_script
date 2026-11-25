// ============================================================================
// farmer.test.ts — 运行器：出售规则、设置解析，以及对一个假游戏服务器的完整流程
// ============================================================================

import { describe, expect, it } from 'vitest'
import { ElevatorClient } from '../src/client.ts'
import type { FetchLike } from '../src/client.ts'
import { Farmer, selectLootToSell } from '../src/farmer.ts'
import type { LogLevel } from '../src/farmer.ts'
import { makeBuild } from '../src/model.ts'
import { DEFAULT_SETTINGS, normalizeSettings, parseLoadout } from '../src/settings.ts'
import type { Settings } from '../src/settings.ts'
import { load, remove } from '../src/storage.ts'
import type { Json } from '../src/types.ts'

it('出售规则：红 / 金每种留一件最值钱的，紫 / 绿 / 白只卖本局新增', () => {
    const stock = [
        { stock_id: 1, id: 'g01', kind: 'collection', rarity: 'gold', value: 5000, quantity: 1 }, // 旧
        { stock_id: 2, id: 'g01', kind: 'collection', rarity: 'gold', value: 7000, quantity: 2 }, // 最好
        { stock_id: 3, id: 'r09', kind: 'collection', rarity: 'red', value: 40000 }, // 唯一一件
        { stock_id: 4, id: 'mat', kind: 'material', rarity: 'white', value: 100, quantity: 3 },
        { stock_id: 5, id: 'key', kind: 'material', rarity: 'blue', value: 900 }, // 蓝色保留
        { stock_id: 6, id: 'coffee', kind: 'supply', rarity: 'green', value: 300 }, // 补给保留
        { stock_id: 7, id: 'crate', kind: 'crate', rarity: 'gold', value: 9000 }, // 箱子保留
        { stock_id: 8, id: 'old', kind: 'material', rarity: 'purple', value: 50 }, // 开局前已有
        { stock_id: 9, id: 'pie', kind: 'material', rarity: 'purple', value: 800 } // 新
    ]
    const picked = selectLootToSell(
        stock,
        new Set([1, 8]),
        new Set(['red', 'gold']),
        new Set(['purple', 'green', 'white'])
    )
    const got = new Set(picked.map(([it, q]) => `${it.stock_id}x${q}`))
    expect(got).toEqual(new Set(['2x1', '1x1', '4x3', '9x1']))
})

describe('设置', () => {
    it('解析携带物资', () => {
        expect(parseLoadout('field_surgery=12, coffee=10,pouch')).toEqual({ field_surgery: 12, coffee: 10, pouch: 1 })
        expect(() => parseLoadout('coffee=abc')).toThrow(/携带物资格式错误/)
    })

    it('丢弃类型不符的已存设置', () => {
        const s = normalizeSettings({ mode: 'real', runs: '5', map: 'nowhere', maxFloor: -3, delay: 0.2, extra: 1 })
        expect(s.mode).toBe('real')
        expect(s.runs).toBe(DEFAULT_SETTINGS.runs)
        expect(s.map).toBe('mall')
        expect(s.maxFloor).toBeNull()
        expect(s.delay).toBe(0.2)
        expect('extra' in s).toBe(false)
    })

    it('旧版默认携带物资升级为新默认值，用户改过的保持不变', () => {
        for (const old of [
            'adrenaline=2,pouch=1,field_surgery=50,triage_vial=20,coffee=10,grenade=2',
            'adrenaline=2,ice_tea=50,coffee=50,triage_vial=50,slow_serum=50,barrier_serum=50,critical_serum=50,field_surgery=50'
        ])
            expect(normalizeSettings({ loadout: old }).loadout).toBe(DEFAULT_SETTINGS.loadout)
        expect(DEFAULT_SETTINGS.loadout).toMatch(/field_surgery=50/)
        expect(DEFAULT_SETTINGS.loadout).toMatch(/smoke=50/)
        expect(DEFAULT_SETTINGS.loadout).not.toMatch(/pouch|doll/)
        expect(normalizeSettings({ loadout: 'coffee=5' }).loadout).toBe('coffee=5')
    })
})

// ---------------------------------------------------------------------------
// 假游戏服务器：练习局，每次连战下 10 层，F21 起只能撤离；F11 有一个首领
// ---------------------------------------------------------------------------
function fakeServer(opts: { cooldownOnce?: boolean } = {}) {
    let rev = 0
    let cooldownPending = Boolean(opts.cooldownOnce)
    const commands: string[] = []
    const state: Json = {
        revision: 'r0',
        player: {
            gold: 0,
            tickets: 0,
            paid_tickets: 0,
            rush_floors: 10,
            rush_skip_valuable: true,
            loadout: {},
            secondary: { crit_multiplier: 1.9, pierce: 0.26, leech: 0.06 }
        },
        atlas: { known: 1, total: 2, items: [], rainbow: [{ name: '虹彩之证', unlocked: false }] },
        warehouse: [
            { stock_id: 1, id: 'g01', name: '金', kind: 'collection', rarity: 'gold', value: 5000, quantity: 1 },
            { stock_id: 2, id: 'g01', name: '金', kind: 'collection', rarity: 'gold', value: 7000, quantity: 1 },
            { stock_id: 3, id: 'p01', name: '紫', kind: 'material', rarity: 'purple', value: 800, quantity: 4 }
        ],
        run: null,
        controls: []
    }
    const sold: Json[] = []
    const starts: Json[] = []
    const runControls = (floor: number) =>
        (floor >= 21 ? ['extract'] : ['rush', 'down', 'extract']).map(c => ({ command: c, params: {}, label: c }))
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

    const fetch: FetchLike = async (url, init) => {
        if (url.endsWith('/game/state')) return json({ state, sync_token: `t${rev}` })
        const { command, params } = JSON.parse(String(init.body))
        if (cooldownPending && command === 'rush') {
            cooldownPending = false
            return json({ error: { code: 'cooldown', message: '操作太快' } }, 429)
        }
        commands.push(command)
        let result: Json = { outcome: 'ongoing', summary: '' }
        if (command === 'simulate' || command === 'start') {
            if (command === 'start') starts.push(params)
            state.run = {
                map: 'mall',
                floor: 1,
                hp: 666,
                max_hp: 666,
                shield: 0,
                simulation: command === 'simulate',
                effective_stats: { hp: 666, atk: 150, def: 70, agi: 77, luk: 21 },
                event: { kind: 'ready' },
                bag: [{ id: 'coffee', index: 1, name: '冰美式' }], // 还有回复：不会因「回复用完」提前撤离
                safe: [],
                safe_capacity: 4,
                field: [],
                field_slots: [null, null]
            }
        } else if (command === 'rush') {
            state.run.floor += 10
            if (state.run.floor === 11) {
                result = {
                    outcome: 'ongoing',
                    summary: '连战 10 层',
                    rush: { floors: [{ loot: [{ id: 'prism_a', name: '虹彩之证', rarity: 'rainbow' }] }] }
                }
            }
        } else if (command === 'extract') {
            state.run = null
            result = { outcome: 'extracted', summary: '撤离成功' }
        } else if (command === 'sell') {
            sold.push(params)
            state.warehouse = state.warehouse.filter((it: Json) => it.stock_id !== params.stock_id)
        } else {
            return json({ error: { code: 'invalid', message: `未知命令 ${command}` } }, 400)
        }
        rev += 1
        state.revision = `r${rev}`
        state.controls = state.run ? runControls(state.run.floor) : []
        return json({ result, state, sync_token: `t${rev}` })
    }
    return { fetch, commands, state, sold, starts }
}

function makeFarmer(server: ReturnType<typeof fakeServer>, settings: Partial<Settings> = {}) {
    const logs: [string, LogLevel][] = []
    const farmer = new Farmer(
        { ...DEFAULT_SETTINGS, delay: 0, ...settings },
        { log: (m, l = 'info') => logs.push([m, l]), progress: () => {} },
        new ElevatorClient('https://example.test/elevator/api', server.fetch)
    )
    farmer.policy.affixBuild = makeBuild({ hp: 666, atk: 150, defense: 70, agi: 77, luk: 21, critMult: 1.9 })
    farmer.policy.loadoutKey = JSON.stringify([])
    return { farmer, logs }
}

describe('完整流程（假服务器）', () => {
    it('开练习局、连战、撤离，记录彩虹掉落与历史', async () => {
        remove('history')
        const server = fakeServer({ cooldownOnce: true })
        const { farmer, logs } = makeFarmer(server, { runs: 2 })
        await farmer.run()
        // 两局：每局 simulate → rush → rush → extract（第一次 rush 遇到限速后重试成功）
        expect(server.commands).toEqual(['simulate', 'rush', 'rush', 'extract', 'simulate', 'rush', 'rush', 'extract'])
        expect(logs.some(([m, l]) => l === 'good' && m.includes('彩虹掉落：虹彩之证'))).toBe(true)
        const history = load<Json[]>('history', [])
        expect(history).toHaveLength(2)
        expect(history[0]).toMatchObject({
            mode: 'sim',
            map: 'mall',
            floor: 21,
            outcome: 'extracted',
            rainbows: ['虹彩之证']
        })
        expect(farmer.trace).toHaveLength(6)
    })

    it('仓库去重：只卖重复的红 / 金，不开局、不卖紫色', async () => {
        const server = fakeServer()
        const { farmer } = makeFarmer(server, { mode: 'real' })
        await farmer.run('dedupe')
        expect(server.sold).toEqual([{ stock_id: 1, quantity: 1 }])
        expect(server.commands).toEqual(['sell'])
    })

    it('「立即暂停」：停在当前步骤，对局保留', async () => {
        remove('history')
        const server = fakeServer()
        const { farmer, logs } = makeFarmer(server, { runs: 3 })
        const origLog = logs.push.bind(logs)
        logs.push = (...items) => {
            if (items.some(([m]) => m.includes('局开始'))) farmer.requestStopNow()
            return origLog(...items)
        }
        await farmer.run()
        expect(server.commands).toEqual(['simulate'])
        expect(server.state.run).not.toBeNull()
        expect(logs.some(([m]) => m.includes('已暂停'))).toBe(true)
        expect(load<Json[]>('history', [])).toHaveLength(0)
    })

    it('正式下潜带上开局页选定的两套追寻祝福', async () => {
        const server = fakeServer()
        server.state.player.tickets = 1
        server.state.heirlooms = { focus: ['B06', 'B08'], needs_focus: false, owned: {}, pieces: {}, sets: {} }
        const { farmer } = makeFarmer(server, { mode: 'real', runs: 1, restock: false })
        await farmer.run()
        expect(server.commands[0]).toBe('start')
        expect(server.starts[0]).toMatchObject({ map_id: 'mall', focus: ['B06', 'B08'] })
    })

    it('尚未选择追寻祝福时不开正式局', async () => {
        const server = fakeServer()
        server.state.player.tickets = 1
        server.state.heirlooms = { focus: [], needs_focus: false, owned: {}, pieces: {}, sets: {} }
        const { farmer, logs } = makeFarmer(server, { mode: 'real', runs: 1, restock: false })
        await farmer.run()
        expect(server.commands).toEqual([])
        expect(logs.some(([m, l]) => l === 'warn' && m.includes('追寻祝福'))).toBe(true)
    })

    it('练习模式不会动进行中的正式局', async () => {
        const server = fakeServer()
        server.state.run = { floor: 50, simulation: false }
        const { farmer, logs } = makeFarmer(server)
        await farmer.run()
        expect(server.commands).toEqual([])
        expect(logs.some(([m, l]) => l === 'warn' && m.includes('进行中的正式局'))).toBe(true)
    })

    it('未登录时提示先登录', async () => {
        const fetch: FetchLike = async () =>
            new Response(JSON.stringify({ error: { code: 'sign_in_required', message: '请登录后继续' } }), {
                status: 401
            })
        const logs: string[] = []
        const farmer = new Farmer(
            DEFAULT_SETTINGS,
            { log: m => logs.push(m), progress: () => {} },
            new ElevatorClient('x', fetch)
        )
        await farmer.run()
        expect(logs.some(m => m.includes('请先在游戏页面登录'))).toBe(true)
    })
})
