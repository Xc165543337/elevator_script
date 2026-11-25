// ============================================================================
// gear.ts — 装备 → Build（移植自 gear.py）
//
// 实测：不同装备上的相同词条不叠加（3 件破甲仍显示 26% 穿透），每个词条只计一次；
// 套装加成在此基础上另加。
// ============================================================================

import { makeBuild } from './model.ts'
import type { Build } from './model.ts'
import type { Json } from './types.ts'

export const SLOTS = ['武器', '护甲', '挂件', '核心'] as const
const STAT_KEYS: Record<string, string> = { 生命: 'hp', 攻击: 'atk', 防御: 'def', 敏捷: 'agi', 幸运: 'luk' }
/** 词条 id → (Build 字段, 单件数值) */
const AFFIX_EFFECT: Record<string, [string, number]> = {
    pierce: ['pierce', 0.26],
    vampire: ['vampire', 0.1],
    heavy_strike: ['heavy', 0.38],
    berserk: ['berserk', 0.075],
    thorns: ['thorns', 0.25],
    shield_start: ['startShield', 25]
}
const FLAG_AFFIXES: Record<string, 'desperate' | 'lastStand'> = { wager: 'desperate', last_stand: 'lastStand' }
/** 套装加成短语 → Build 字段（百分数按小数计） */
const SET_SPECIAL: Record<string, string> = {
    命中吸血: 'leech',
    追击率: 'combo',
    越战越勇: 'berserk',
    对首领伤害: 'bossBonus',
    反甲刺猬: 'thorns',
    出门套盾: 'startShield'
}

export interface Item {
    equipId: number
    name: string
    slot: string
    rarity: string
    setName: string | null
    stats: [string, number][]
    affixes: string[]
    affixNames: string[]
}

export function itemFromState(e: Json): Item {
    const stats = Object.entries((e.stats ?? {}) as Record<string, number>).sort(([a], [b]) =>
        a < b ? -1 : a > b ? 1 : 0
    )
    const affixes = (e.affixes ?? []) as Json[]
    return {
        equipId: e.equip_id,
        name: e.name,
        slot: e.slot,
        rarity: e.rarity,
        setName: e.set_name || null,
        stats,
        affixes: affixes.map(a => a.id),
        affixNames: affixes.map(a => a.name)
    }
}

export type SetBonuses = Record<string, Record<number, Record<string, number>>>

/** 从装备描述解析套装加成：{套装名: {2: {...}, 4: {...}}} */
export function parseSetBonuses(equipment: Json[]): SetBonuses {
    const sets: SetBonuses = {}
    for (const e of equipment) {
        const name = e.set_name
        if (!name || name in sets) continue
        const tiers: Record<number, Record<string, number>> = {}
        for (const [, pieces, body] of String(e.description ?? '').matchAll(/(\d)件 · ([^\n]+)/g)) {
            const bonus: Record<string, number> = {}
            for (const [, key, value, pct] of body.matchAll(/([一-龥]+)\+(\d+(?:\.\d+)?)(%?)/g)) {
                const field = STAT_KEYS[key] ?? SET_SPECIAL[key]
                if (field) bonus[field] = (bonus[field] ?? 0) + (pct ? Number(value) / 100 : Number(value))
            }
            tiers[Number(pieces)] = bonus
        }
        sets[name] = tiers
    }
    return sets
}

/** 不含任何已穿装备与套装加成的永久属性 */
export function baseStats(player: Json, gearSets: Set<string>): Record<string, number> {
    const base: Record<string, number> = {}
    for (const [stat, sources] of Object.entries((player.stat_sources ?? {}) as Record<string, Json[]>)) {
        let total = 0
        for (const src of sources) {
            const label = String(src.source)
            if (label.includes('#') || [...gearSets].some(s => label.startsWith(`${s} ·`))) continue
            if (src.op === 'base' || src.op === 'add') total += src.value
        }
        base[stat] = total
    }
    return base
}

export class GearModel {
    readonly items: Item[]
    readonly sets: SetBonuses
    readonly base: Record<string, number>
    readonly current: Item[]

    constructor(state: Json) {
        const equipment = (state.equipment ?? []) as Json[]
        this.items = equipment.map(itemFromState)
        this.sets = parseSetBonuses(equipment)
        this.base = baseStats(state.player, new Set(Object.keys(this.sets)))
        const loadout = (state.player.loadout ?? {}) as Record<string, number>
        const byId = new Map(this.items.map(i => [i.equipId, i]))
        this.current = SLOTS.map(s => {
            const item = byId.get(loadout[s])
            if (!item) throw new Error(`装备栏「${s}」的装备 #${loadout[s]} 不在装备列表中`)
            return item
        })
    }

    build(items: Item[], label = ''): Build {
        const stats: Record<string, number> = { ...this.base }
        const extra: Record<string, number> = {}
        const add = (k: string, v: number) => (extra[k] = (extra[k] ?? 0) + v)
        for (const it of items) for (const [k, v] of it.stats) stats[k] = (stats[k] ?? 0) + v
        const setCounts = new Map<string, number>()
        for (const i of items) if (i.setName) setCounts.set(i.setName, (setCounts.get(i.setName) ?? 0) + 1)
        for (const [setName, count] of setCounts) {
            for (const [pieces, bonus] of Object.entries(this.sets[setName] ?? {})) {
                if (count < Number(pieces)) continue
                for (const [k, v] of Object.entries(bonus)) {
                    if (k in stats) stats[k] += v
                    else add(k, v)
                }
            }
        }
        const flags: Partial<Build> = {}
        let critMult = 1.6
        for (const a of new Set(items.flatMap(it => it.affixes))) {
            if (a in AFFIX_EFFECT) add(...AFFIX_EFFECT[a])
            else if (a in FLAG_AFFIXES) flags[FLAG_AFFIXES[a]] = true
            else if (a === 'high_roller') critMult = 1.9
            else if (a === 'greed') add('greed', 0.2) // 贪婪诅咒：承受伤害 +20%
        }
        const x = (k: string) => extra[k] ?? 0
        return makeBuild({
            hp: stats.hp ?? 0,
            atk: stats.atk ?? 0,
            defense: stats.def ?? 0,
            agi: stats.agi ?? 0,
            luk: stats.luk ?? 0,
            pierce: x('pierce'),
            leech: x('leech'),
            combo: x('combo'),
            heavy: x('heavy'),
            berserk: x('berserk'),
            vampire: x('vampire'),
            thorns: x('thorns'),
            bossBonus: x('bossBonus'),
            startShield: x('startShield'),
            critMult,
            takenMult: 1.0 + x('greed'),
            label,
            ...flags
        })
    }
}
