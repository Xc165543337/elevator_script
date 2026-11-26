// ============================================================================
// settings.ts — 可在面板中配置的参数（对应 farm.py 的命令行选项）
// ============================================================================

export const MAPS: [string, string][] = [
    ['mall', '废弃商场（刷首领最佳）'],
    ['subway', '地铁'],
    ['abyss', '深渊'],
    ['auction', '拍卖行'],
    ['tide', '潮汐发电站（归潮档案）'],
    ['zenith', '天顶']
]

/** 每种补给单次开局最多携带的数量（服务器限制） */
export const MAX_CARRY = 50
/** 商店能买到的回复药与烟雾弹带满，外加 2 支肾上腺素；手雷、护盾需合成（消耗材料），替身娃娃与扩容袋留在家里 */
export const DEFAULT_LOADOUT = [
    'adrenaline=2',
    ...[
        'coffee',
        'ice_tea',
        'triage_vial',
        'barrier_serum',
        'slow_serum',
        'critical_serum',
        'field_surgery',
        'smoke'
    ].map(id => `${id}=${MAX_CARRY}`)
].join(',')

export interface Settings {
    /** sim = 练习（免费、无收益）；real = 正式（消耗门票与金币） */
    mode: 'sim' | 'real'
    /** threshold = 手调规则（policy.ts）；markov = 求解的马尔可夫决策过程（markov.ts，后台求解约 5 分钟） */
    policy: 'threshold' | 'markov'
    map: string
    runs: number
    /** 到达即撤离；null = 不限 */
    maxFloor: number | null
    loadout: string
    restock: boolean
    /** 模型判断同样安全时，连战开启「自动打精英」（每局停顿更少） */
    rushElites: boolean
    buyTickets: boolean
    goldReserve: number
    /** 两次请求之间的最短间隔（秒） */
    delay: number
    balanceGearScore: number
    sellLoot: boolean
    sellRarities: string
    dedupeRarities: string
    verbose: boolean
}

export const DEFAULT_SETTINGS: Settings = {
    mode: 'sim',
    policy: 'threshold',
    map: 'mall',
    runs: 1,
    maxFloor: null,
    loadout: DEFAULT_LOADOUT,
    restock: true,
    rushElites: true,
    buyTickets: false,
    goldReserve: 1_000_000,
    delay: 0.05,
    balanceGearScore: -1.0,
    sellLoot: false,
    sellRarities: 'purple,green,white',
    dedupeRarities: 'red,gold',
    verbose: false
}

/** 旧版本的默认携带物资：保存的值仍是旧默认值（用户没改过）时换成新默认值 */
const OLD_DEFAULT_LOADOUTS = new Set([
    'adrenaline=2,pouch=1,field_surgery=50,triage_vial=20,coffee=10,grenade=2',
    'adrenaline=2,ice_tea=50,coffee=50,triage_vial=50,slow_serum=50,barrier_serum=50,critical_serum=50,field_surgery=50'
])

/** 合并已保存的设置，丢弃类型不符的字段（旧版本或手动改坏的数据） */
export function normalizeSettings(saved: unknown): Settings {
    const out: Settings = { ...DEFAULT_SETTINGS }
    if (!saved || typeof saved !== 'object') return out
    const src = saved as Record<string, unknown>
    for (const key of Object.keys(DEFAULT_SETTINGS) as (keyof Settings)[]) {
        const value = src[key]
        const def = DEFAULT_SETTINGS[key]
        if (key === 'maxFloor') {
            if (value === null || (typeof value === 'number' && Number.isInteger(value) && value > 0))
                out.maxFloor = value
        } else if (key === 'mode') {
            if (value === 'sim' || value === 'real') out.mode = value
        } else if (key === 'policy') {
            if (value === 'threshold' || value === 'markov') out.policy = value
        } else if (key === 'map') {
            if (typeof value === 'string' && MAPS.some(([id]) => id === value)) out.map = value
        } else if (typeof value === typeof def && (typeof value !== 'number' || Number.isFinite(value))) {
            ;(out as unknown as Record<string, unknown>)[key] = value
        }
    }
    if (OLD_DEFAULT_LOADOUTS.has(out.loadout)) out.loadout = DEFAULT_LOADOUT
    return out
}

/** "field_surgery=12,coffee=10" → {field_surgery: 12, coffee: 10}；数量省略时为 1 */
export function parseLoadout(text: string): Record<string, number> {
    const out: Record<string, number> = {}
    for (const part of text.split(',').map(p => p.trim())) {
        if (!part) continue
        const [rawKey, rawQty = ''] = part.split('=', 2)
        const key = rawKey.trim()
        const qty = rawQty.trim() === '' ? 1 : Number(rawQty.trim())
        if (!key || !Number.isInteger(qty) || qty < 0)
            throw new Error(`携带物资格式错误：「${part}」（应为 物品id=数量）`)
        out[key] = qty
    }
    return out
}

export function parseList(text: string): Set<string> {
    return new Set(
        text
            .split(',')
            .map(s => s.trim())
            .filter(Boolean)
    )
}
