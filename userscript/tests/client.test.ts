// ============================================================================
// client.test.ts — API 客户端：请求头、增量合并、错误分类
// ============================================================================

import { describe, expect, it } from 'vitest'
import { ElevatorClient, GameError, NetworkError } from '../src/client.ts'
import type { FetchLike } from '../src/client.ts'
import type { Json } from '../src/types.ts'

interface Call {
    url: string
    init: RequestInit
}

function fakeFetch(responses: (Json | (() => Response | Promise<Response>))[]): { fetch: FetchLike; calls: Call[] } {
    const calls: Call[] = []
    const fetch: FetchLike = async (url, init) => {
        calls.push({ url, init })
        const next = responses.shift()
        if (typeof next === 'function') return next()
        return new Response(JSON.stringify(next), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    return { fetch, calls }
}

const ROOT = 'https://example.test/elevator/api'
const headers = (c: Call) => c.init.headers as Record<string, string>

describe('ElevatorClient', () => {
    it('拉取状态：同源会话 Cookie + 同步头', async () => {
        const { fetch, calls } = fakeFetch([{ state: { revision: 'r1', run: null }, sync_token: 't1' }])
        const client = new ElevatorClient(ROOT, fetch)
        await client.refresh()
        expect(calls[0].url).toBe(`${ROOT}/game/state`)
        expect(calls[0].init.method).toBe('GET')
        expect(calls[0].init.credentials).toBe('same-origin')
        expect(headers(calls[0])['X-Elevator-Sync']).toBe('1')
        expect(headers(calls[0])['X-Elevator-Base']).toBeUndefined()
        expect(client.revision).toBe('r1')
        expect(client.syncToken).toBe('t1')
    })

    it('命令携带 revision 与基准令牌，并合并增量', async () => {
        const state = {
            revision: 'r1',
            player: { gold: 10, tickets: 3 },
            bag: [
                { k: 1, v: 'a' },
                { k: 2, v: 'b' }
            ]
        }
        const delta = {
            base: 't1',
            base_revision: 'r1',
            fields: { revision: 'r2' },
            sections: { player: { gold: 7 } },
            lists: {
                bag: {
                    key: 'k',
                    upsert: [
                        { k: 2, v: 'B' },
                        { k: 3, v: 'c' }
                    ],
                    order: [3, 2]
                }
            }
        }
        const { fetch, calls } = fakeFetch([
            { state, sync_token: 't1' },
            { result: { ok: true }, delta, sync_token: 't2' }
        ])
        const client = new ElevatorClient(ROOT, fetch)
        await client.refresh()
        const data = await client.command('buy', { id: 'coffee' })
        expect(data.result).toEqual({ ok: true })
        expect(JSON.parse(String(calls[1].init.body))).toEqual({
            command: 'buy',
            params: { id: 'coffee' },
            revision: 'r1'
        })
        expect(headers(calls[1])['X-Elevator-Base']).toBe('t1')
        expect(client.state.revision).toBe('r2')
        expect(client.state.player).toEqual({ gold: 7, tickets: 3 })
        expect(client.state.bag).toEqual([
            { k: 3, v: 'c' },
            { k: 2, v: 'B' }
        ])
        expect(client.syncToken).toBe('t2')
    })

    it('增量基准不一致时重新拉取完整状态', async () => {
        const { fetch, calls } = fakeFetch([
            { state: { revision: 'r1' }, sync_token: 't1' },
            { result: {}, delta: { base: 'other', base_revision: 'r1', fields: {} }, sync_token: 't2' },
            { state: { revision: 'r9' }, sync_token: 't9' }
        ])
        const client = new ElevatorClient(ROOT, fetch)
        await client.refresh()
        await client.command('down')
        expect(calls.map(c => c.url.replace(ROOT, ''))).toEqual(['/game/state', '/game/command', '/game/state'])
        expect(client.revision).toBe('r9')
    })

    it('服务器错误 → GameError（带错误码），并吸收附带的状态', async () => {
        const { fetch } = fakeFetch([
            () =>
                new Response(
                    JSON.stringify({ error: { code: 'invalid', message: '不能这样做' }, state: { revision: 'r5' } }),
                    {
                        status: 400
                    }
                )
        ])
        const client = new ElevatorClient(ROOT, fetch)
        const err = await client.command('fight').catch(e => e)
        expect(err).toBeInstanceOf(GameError)
        expect(err.code).toBe('invalid')
        expect(client.revision).toBe('r5')
    })

    it('未登录 → GameError(sign_in_required)', async () => {
        const { fetch } = fakeFetch([
            () =>
                new Response(JSON.stringify({ error: { code: 'sign_in_required', message: '请登录后继续' } }), {
                    status: 401
                })
        ])
        const err = await new ElevatorClient(ROOT, fetch).refresh().catch(e => e)
        expect(err).toBeInstanceOf(GameError)
        expect(err.code).toBe('sign_in_required')
    })

    it('HTML 错误页与断网 → NetworkError', async () => {
        const html = fakeFetch([() => new Response('<html>502</html>', { status: 502 })])
        expect(await new ElevatorClient(ROOT, html.fetch).refresh().catch(e => e)).toBeInstanceOf(NetworkError)
        const down: FetchLike = async () => {
            throw new TypeError('Failed to fetch')
        }
        expect(await new ElevatorClient(ROOT, down).refresh().catch(e => e)).toBeInstanceOf(NetworkError)
    })
})
