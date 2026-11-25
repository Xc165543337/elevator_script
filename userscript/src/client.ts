// ============================================================================
// client.ts — 游戏 Web API 客户端（移植自 client.py）
//
// 脚本运行在游戏自己的页面上：请求与页面同源，浏览器自动带上玩家已登录的会话 Cookie，
// 因此不需要账号密码，也不受跨域限制。
// ============================================================================

import type { Json } from './types.ts'

export class GameError extends Error {
    constructor(
        readonly code: string | null,
        message: string,
        readonly payload: Json = {}
    ) {
        super(`${code}: ${message}`)
        this.name = 'GameError'
    }
}

/** 网络故障、网关错误页等（对应 Python 的 requests.RequestException / ValueError） */
export class NetworkError extends Error {
    constructor(message: string) {
        super(message)
        this.name = 'NetworkError'
    }
}

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>

export class ElevatorClient {
    state: Json = null
    syncToken = ''

    constructor(
        readonly apiRoot: string = `${location.origin}/elevator/api`,
        private readonly fetchFn: FetchLike = (url, init) => fetch(url, init),
        readonly timeoutMs = 20_000
    ) {}

    private async call(path: string, body?: unknown): Promise<Json> {
        const headers: Record<string, string> = { Accept: 'application/json' }
        if (path.startsWith('/game/')) {
            headers['X-Elevator-Sync'] = '1'
            if (body !== undefined && this.syncToken) headers['X-Elevator-Base'] = this.syncToken
        }
        if (body !== undefined) headers['Content-Type'] = 'application/json'
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), this.timeoutMs)
        let resp: Response
        try {
            resp = await this.fetchFn(this.apiRoot + path, {
                method: body === undefined ? 'GET' : 'POST',
                headers,
                body: body === undefined ? undefined : JSON.stringify(body),
                credentials: 'same-origin',
                signal: controller.signal
            })
        } catch (err) {
            throw new NetworkError(`请求失败（${(err as Error).message || err}）`)
        } finally {
            clearTimeout(timer)
        }
        let data: Json
        try {
            data = await resp.json()
        } catch {
            // HTML 错误页 / 截断的响应（网关抖动、服务器重启）
            throw new NetworkError(`非 JSON 响应（HTTP ${resp.status}）`)
        }
        if (!resp.ok || data?.error) {
            const err = data?.error
            const code = err && typeof err === 'object' ? err.code : (data?.code ?? null)
            const msg = err && typeof err === 'object' ? err.message : String(err ?? data?.detail ?? '')
            if (data?.state && typeof data.state === 'object') this.setState(data.state, data.sync_token)
            throw new GameError(code ?? null, msg || `请求失败（HTTP ${resp.status}）`, data)
        }
        return data
    }

    private setState(state: Json, token?: string | null): void {
        this.state = state
        if (token) this.syncToken = token
    }

    async refresh(): Promise<Json> {
        const data = await this.call('/game/state')
        this.setState(data.state ?? data, data.sync_token)
        return this.state
    }

    get revision(): Json {
        return this.state?.revision || this.state?.version
    }

    /** 结构化游戏命令（start、down、fight、flee、extract、use、event……） */
    async command(command: string, params: Record<string, unknown> = {}): Promise<Json> {
        const data = await this.call('/game/command', { command, params, revision: this.revision })
        await this.absorb(data)
        return data
    }

    /** 合并服务器返回的增量；基准不一致时重新拉取完整状态 */
    private async absorb(data: Json): Promise<void> {
        if (data.delta) {
            const patch = data.delta
            const next =
                patch.base === this.syncToken && patch.base_revision === this.revision ? this.merge(patch) : null
            if (next) this.setState(next, data.sync_token)
            else await this.refresh()
        } else if (data.state) {
            this.setState(data.state, data.sync_token)
        }
    }

    /** 应用增量；列表顺序引用了未知条目时返回 null（改为重新拉取完整状态） */
    private merge(patch: Json): Json | null {
        const next: Json = { ...this.state, ...patch.fields }
        for (const [key, changed] of Object.entries(patch.sections || {})) {
            next[key] = { ...next[key], ...(changed as object) }
        }
        for (const [name, upd] of Object.entries(patch.lists || {}) as [string, Json][]) {
            const original: Json[] = this.state[name] || []
            const byId = new Map(original.map(row => [row[upd.key], row]))
            for (const row of upd.upsert || []) byId.set(row[upd.key], row)
            const order: unknown[] = upd.order || original.map(row => row[upd.key])
            if (!order.every(i => byId.has(i))) return null
            next[name] = order.map(i => byId.get(i))
        }
        return next
    }
}
