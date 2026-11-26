// ============================================================================
// field.ts — 局内道具：服务器规格及其对实时战斗的影响（移植自 field.py）
//
// 局内状态里，部分局内修正已经折算进 effective_stats（攻击 / 防御 / 敏捷）和 player.secondary
// （穿透 / 追击 / 吸血 / 首领增伤 / 打击减伤）；闪避、暴击、首击加成、反伤和防护 / 蓄力次数只以道具编号出现，
// 这里按服务器规格（data/field_items.json，导出自游戏 content.json）重建。
// ============================================================================

import FIELD_DATA from '../../data/field_items.json'
import type { Build } from './model.ts'
import type { Json } from './types.ts'

export interface FieldSpec {
    name: string
    mode: string
    mods?: Record<string, number>
    hits?: number
    reduction?: number
    mult?: number
    sure?: boolean
    blood?: boolean
    judgement?: boolean
    cost_hp?: number
    heal?: number
    shield?: number
    [key: string]: unknown
}

export const SPECS: Record<string, FieldSpec> = (FIELD_DATA as Json).items
const MOD_CAPS: Record<string, [number, number]> = (FIELD_DATA as Json).mod_caps
/** 服务器 FIELD_TIMED_SLOTS：限层增益最多同时 3 个 */
export const TIMED_SLOTS = 3
const FIGHT_MODES = new Set(['timed', 'guard', 'burst'])

function cap(key: string, value: number): number {
    const [lo, hi] = MOD_CAPS[key] ?? [-Infinity, Infinity]
    return Math.min(hi, Math.max(lo, value))
}

function addMods(totals: Record<string, number>, entries: Json[]): Record<string, number> {
    for (const entry of entries) {
        for (const [key, value] of Object.entries(SPECS[entry?.id ?? '']?.mods ?? {})) {
            totals[key] = (totals[key] ?? 0) + value
        }
    }
    return totals
}

/** 佩戴的临装与生效中的限层增益的修正合计（未封顶，服务器顺序） */
export function activeMods(run: Json): Record<string, number> {
    return addMods({}, [...(run.field_buffs || []), ...(run.field_slots || [])])
}

/** 只算佩戴的临装（整局有效） */
export function wornMods(run: Json): Record<string, number> {
    return addMods({}, run.field_slots || [])
}

/**
 * 补上状态里没有数值化的局内效果。mitigationInStats：服务器的二级属性已含打击减伤（较新的服务器），
 * 其中已包括局内道具。
 */
export function liveBuild(run: Json, build: Build, mitigationInStats = false): Build {
    const mods = activeMods(run)
    const extraReduction = mitigationInStats ? 0 : cap('mitigation', mods.mitigation ?? 0)
    let out: Build = {
        ...build,
        dodgeBonus: cap('dodge', mods.dodge ?? 0),
        critBonus: build.critBonus + cap('crit', mods.crit ?? 0),
        reduction: build.reduction + extraReduction,
        heavy: build.heavy + cap('first', mods.first ?? 0),
        thorns: build.thorns + cap('thorns', mods.thorns ?? 0)
    }
    for (const charge of (run.field_charges || []) as Json[]) {
        const spec = SPECS[charge.id ?? '']
        const left = Number(charge.left || 0)
        if (spec?.mode === 'guard') out = { ...out, guardHits: left, guardReduction: spec.reduction ?? 0 }
        else if (spec?.mode === 'burst' && !spec.blood) {
            out = { ...out, burstHits: left, burstMult: spec.mult ?? 1, burstSure: Boolean(spec.sure) }
        }
    }
    return out
}

/** 会改变这场战斗的局内道具（回复类除外） */
export interface FieldPrep {
    id: string
    name: string
    spec: FieldSpec
}

/** 按服务器的发动规则，当前可用的限层增益 / 防护 / 蓄力道具 */
export function fightPreparations(run: Json): FieldPrep[] {
    const buffs = new Set(((run.field_buffs || []) as Json[]).map(b => b.id))
    const charges = (run.field_charges || []) as Json[]
    if (charges.some(c => SPECS[c.id ?? '']?.blood)) return [] // 锁血蓄力期间禁止回复与护盾
    const charged = new Set(charges.map(c => SPECS[c.id ?? '']?.mode))
    const out: FieldPrep[] = []
    for (const item of (run.field || []) as Json[]) {
        const spec = SPECS[item.id ?? '']
        if (!spec || !item.quantity || !FIGHT_MODES.has(spec.mode)) continue
        if (spec.blood || spec.judgement || spec.cost_hp) continue // 孤注一掷类道具交给玩家手动
        if (charged.has(spec.mode) || buffs.has(item.id)) continue
        if (spec.mode === 'timed' && buffs.size >= TIMED_SLOTS) continue
        out.push({ id: item.id, name: item.name || spec.name, spec })
    }
    return out
}

/** 同一场遭遇中，先发动 first 后还能否再发动 second */
export function compatible(first: FieldPrep, second: FieldPrep, run: Json): boolean {
    if (first.id === second.id) return false
    if (first.spec.mode === second.spec.mode && (first.spec.mode === 'guard' || first.spec.mode === 'burst')) {
        return false
    }
    if (first.spec.mode === 'timed' && second.spec.mode === 'timed') {
        return (run.field_buffs || []).length + 2 <= TIMED_SLOTS
    }
    return true
}

/**
 * 发动 prep 后的 Build。mods = 发动前的修正合计（默认取实时值）。百分比修正作用于基础属性；
 * 每项按合计值封顶，与服务器一致。
 */
export function applyPrep(
    run: Json,
    build: Build,
    prep: FieldPrep,
    mods?: Record<string, number>
): [Build, Record<string, number>] {
    const totals = { ...(mods ?? activeMods(run)) }
    const spec = prep.spec
    if (spec.mode === 'guard')
        return [{ ...build, guardHits: spec.hits ?? 0, guardReduction: spec.reduction ?? 0 }, totals]
    if (spec.mode === 'burst') {
        return [
            { ...build, burstHits: spec.hits ?? 0, burstMult: spec.mult ?? 1, burstSure: Boolean(spec.sure) },
            totals
        ]
    }
    const base = run.stats || run.effective_stats
    const delta: Record<string, number> = {}
    for (const [key, value] of Object.entries(spec.mods ?? {})) {
        const before = totals[key] ?? 0
        delta[key] = cap(key, before + value) - cap(key, before)
        totals[key] = before + value
    }
    const d = (key: string) => delta[key] ?? 0
    return [
        {
            ...build,
            atk: Math.max(0.25 * base.atk, build.atk + base.atk * d('atk')),
            defense: Math.max(0, build.defense + base.def * d('defense')),
            agi: Math.max(0, build.agi + d('agi')),
            leech: Math.min(0.42, build.leech + d('leech')),
            combo: Math.min(0.7, build.combo + d('combo')),
            pierce: Math.min(build.pierceCap, build.pierce + d('pierce')),
            bossBonus: Math.min(1.2, build.bossBonus + d('boss')),
            reduction: build.reduction + d('mitigation'),
            critBonus: build.critBonus + d('crit'),
            dodgeBonus: build.dodgeBonus + d('dodge'),
            heavy: build.heavy + d('first'),
            thorns: Math.min(0.7, build.thorns + d('thorns'))
        },
        totals
    ]
}
