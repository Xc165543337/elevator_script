// ============================================================================
// storage.ts — 本地持久化（设置、校准系数、历史记录）
//
// 脚本以 @grant none 运行在页面环境中，数据存于该站点的 localStorage。
// 隐私窗口、禁用站点数据时读写可能抛错：一律容错，退化为仅本次会话有效。
// ============================================================================

const PREFIX = 'elevator-farmer:'
const memory = new Map<string, string>()

function backend(): Storage | null {
    try {
        return globalThis.localStorage ?? null
    } catch {
        return null
    }
}

export function load<T>(key: string, fallback: T): T {
    let raw: string | null = memory.get(key) ?? null
    try {
        raw = backend()?.getItem(PREFIX + key) ?? raw
    } catch {
        // 读取失败：使用内存副本
    }
    if (raw === null) return fallback
    try {
        return JSON.parse(raw) as T
    } catch {
        return fallback
    }
}

export function save(key: string, value: unknown): void {
    const raw = JSON.stringify(value)
    memory.set(key, raw)
    try {
        backend()?.setItem(PREFIX + key, raw)
    } catch {
        // 写入失败（配额、禁用存储）：仅保留在内存中
    }
}

export function remove(key: string): void {
    memory.delete(key)
    try {
        backend()?.removeItem(PREFIX + key)
    } catch {
        // 忽略
    }
}
