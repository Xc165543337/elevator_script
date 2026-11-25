// ============================================================================
// lock.ts — 防止多个标签页同时操作同一账号（各自的 revision 会互相冲突）
//
// 运行中的标签页定期写入心跳；心跳超过 STALE_MS 视为该标签页已关闭。
// 后台标签页的定时器可能被浏览器延迟到每分钟一次，因此阈值取 3 分钟。
// ============================================================================

import { load, remove, save } from './storage.ts'

const KEY = 'lock'
const STALE_MS = 180_000

interface LockEntry {
    tab: string
    at: number
}

export class TabLock {
    readonly tab = Math.random().toString(36).slice(2)
    private lastBeat = 0

    /** 其他标签页正在运行时返回 false */
    acquire(now = Date.now()): boolean {
        const cur = load<LockEntry | null>(KEY, null)
        if (cur && cur.tab !== this.tab && now - cur.at < STALE_MS) return false
        save(KEY, { tab: this.tab, at: now })
        this.lastBeat = now
        return true
    }

    /** 每步调用；最多每 5 秒写一次 */
    beat(now = Date.now()): void {
        if (now - this.lastBeat < 5_000) return
        save(KEY, { tab: this.tab, at: now })
        this.lastBeat = now
    }

    release(): void {
        const cur = load<LockEntry | null>(KEY, null)
        if (cur?.tab === this.tab) remove(KEY)
    }
}
