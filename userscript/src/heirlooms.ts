// ============================================================================
// heirlooms.ts — 闻岑闪耀祝福（2026-11-25 起，移植自 heirlooms.py）
//
// 祝福装在四个独立槽位；同套两件、四件分别触发套装效果。属性类加成（穿透、暴击、吸血……）已经
// 体现在服务器汇总的二级属性里，这里只补二级属性表达不了的效果。
// ============================================================================

import type { Build } from './model.ts'
import type { Json } from './types.ts'

/** 每套穿戴的件数（套装编号 → 件数） */
export function wornSets(heirlooms: Json | null | undefined): Map<string, number> {
    const pieces: Json = heirlooms?.pieces || {}
    const counts = new Map<string, number>()
    for (const key of (heirlooms?.slots || []) as (string | null)[]) {
        if (!key) continue
        const set: string = pieces[key]?.set || key.split('-')[0]
        counts.set(set, (counts.get(set) ?? 0) + 1)
    }
    return counts
}

/** 二级属性不包含的套装效果对应的 Build 字段（未列出的套装效果尚未建模） */
export function buildEffects(heirlooms: Json | null | undefined): Partial<Build> {
    const b06 = wornSets(heirlooms).get('B06') ?? 0
    const effects: Partial<Build> = {}
    if (b06 >= 2) effects.pierceCap = 0.87 // 缺刻值夜表：穿透在 75% 上限后再加 12 个百分点
    if (b06 >= 4) effects.beat = true // 抢拍：第 4、8、12……回合先手，承受 ×0.65
    return effects
}

/** 服务器的八套祝福（编号与开局页 / 「追寻 1 8」的顺序一致）；面板在读到游戏状态之前就要列出 */
export const BLESSING_SETS: [string, string][] = [
    ['B01', '错线测绘簿'],
    ['B02', '末班灯列'],
    ['B03', '白线压舱契'],
    ['B04', '退件邮路'],
    ['B05', '余温窑账'],
    ['B06', '缺刻值夜表'],
    ['B07', '逆风放线人'],
    ['B08', '雪线留声筒']
]

/** 追寻设置：'' = 沿用游戏内上次的选择，'auto' = 缺件最多的两套，'B03,B08' = 指定两套 */
export const FOCUS_AUTO = 'auto'

function setPieces(heirlooms: Json, set: string): string[] {
    return Object.entries((heirlooms.pieces || {}) as Record<string, Json>)
        .filter(([, spec]) => spec?.set === set)
        .map(([key]) => key)
}

function isOwned(heirlooms: Json, piece: string): boolean {
    return Number((heirlooms.owned || {})[piece] || 0) > 0
}

/** 每套还缺的件数（按服务器顺序） */
export function missingBySet(heirlooms: Json): Map<string, number> {
    const out = new Map<string, number>()
    for (const set of Object.keys(heirlooms.sets || {})) {
        out.set(set, setPieces(heirlooms, set).filter(p => !isOwned(heirlooms, p)).length)
    }
    return out
}

/**
 * 追寻这两套时，一次闪耀掉落是新部件的概率：两套等概率，套内缺的部件权重是已有的两倍
 * （游戏 heirlooms.roll），所以一套 4 件里缺 m 件时为 2m / (m + 4)
 */
export function newDropChance(heirlooms: Json, pair: string[]): number {
    if (!pair.length) return 0
    let total = 0
    for (const set of pair) {
        const parts = setPieces(heirlooms, set)
        const missing = parts.filter(p => !isOwned(heirlooms, p)).length
        const weight = 2 * missing + (parts.length - missing)
        total += weight ? (2 * missing) / weight : 0
    }
    return total / pair.length
}

/** 缺件最多的两套（每次闪耀掉落的新部件最多）；同数时优先已在追寻的套，再按服务器顺序；全部集齐时沿用上次 */
export function autoFocus(heirlooms: Json): string[] {
    const missing = missingBySet(heirlooms)
    const last = (heirlooms.focus || []) as string[]
    if ([...missing.values()].every(m => m === 0) && last.length === 2 && last[0] !== last[1]) return [...last]
    const order = [...missing.keys()]
    const rank = (k: string) => [-(missing.get(k) ?? 0), last.includes(k) ? 0 : 1, order.indexOf(k)]
    return order
        .slice()
        .sort((a, b) => {
            const [x, y] = [rank(a), rank(b)]
            return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]
        })
        .slice(0, 2)
}

/**
 * 正式下潜追寻的两套：服务器没有祝福系统时为 null；没有可用选择时为 undefined。
 * wanted：''（沿用上次）、'auto' 或 'B03,B08'
 */
export function resolveFocus(heirlooms: Json | null | undefined, wanted: string): string[] | null | undefined {
    if (!heirlooms) return null
    const known = Object.keys(heirlooms.sets || {})
    let pair: string[]
    if (wanted === FOCUS_AUTO) pair = autoFocus(heirlooms)
    else if (wanted) pair = wanted.split(',').map(s => s.trim())
    else pair = [...((heirlooms.focus || []) as string[])]
    const valid = pair.length === 2 && pair[0] !== pair[1] && (!known.length || pair.every(k => known.includes(k)))
    return valid ? pair : undefined
}
