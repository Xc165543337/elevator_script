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
