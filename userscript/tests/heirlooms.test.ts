// ============================================================================
// heirlooms.test.ts — 追寻祝福的选择：沿用、自动（缺件最多的两套）、指定（同 Python 版 tests/test_farm.py）
// ============================================================================

import { describe, expect, it } from 'vitest'
import { autoFocus, missingBySet, newDropChance, resolveFocus } from '../src/heirlooms.ts'
import { normalizeSettings, validFocus } from '../src/settings.ts'
import type { Json } from '../src/types.ts'

function collection(owned: Record<string, number>, focus = ['S1', 'S2']): Json {
    const sets: Json = {}
    const pieces: Json = {}
    for (let i = 1; i <= 4; i++) {
        sets[`S${i}`] = { name: `set${i}` }
        for (let j = 1; j <= 4; j++) pieces[`S${i}-${j}`] = { name: `p${i}${j}`, set: `S${i}` }
    }
    return { sets, pieces, owned, focus }
}

describe('追寻祝福', () => {
    it('自动：追寻缺件最多的两套', () => {
        const full = Object.fromEntries([1, 2, 3, 4].map(j => [`S1-${j}`, 1]))
        const coll = collection({ ...full, 'S2-1': 1, 'S2-2': 1, 'S3-1': 1 }) // 缺：S2 2、S3 3、S4 4
        expect(resolveFocus(coll, 'auto')).toEqual(['S4', 'S3'])
    })

    it('自动：同数时保留已在追寻的套，全部集齐时沿用上次', () => {
        const coll = collection({ 'S1-1': 1, 'S2-1': 1, 'S3-1': 1, 'S4-1': 1 }, ['S3', 'S1'])
        expect(autoFocus(coll)).toEqual(['S1', 'S3'])
        const all = Object.fromEntries(Object.keys(collection({}).pieces).map(p => [p, 1]))
        expect(autoFocus(collection(all, ['S2', 'S4']))).toEqual(['S2', 'S4'])
    })

    it('新部件概率与服务器权重一致', () => {
        const coll = collection({ 'S1-1': 1, 'S1-2': 1, 'S1-3': 1 }) // S1 缺 1，S2 缺 4
        expect(newDropChance(coll, ['S1', 'S2'])).toBeCloseTo(0.7, 9) // (2/5 + 1) / 2
        expect(Object.fromEntries(missingBySet(coll))).toEqual({ S1: 1, S2: 4, S3: 4, S4: 4 })
    })

    it('沿用、指定与无效选择', () => {
        const coll = collection({})
        expect(resolveFocus(coll, '')).toEqual(['S1', 'S2'])
        expect(resolveFocus(coll, 'S4,S2')).toEqual(['S4', 'S2'])
        expect(resolveFocus(coll, 'S4,S4')).toBeUndefined()
        expect(resolveFocus(collection({}, []), '')).toBeUndefined()
        expect(resolveFocus(null, 'auto')).toBeNull()
    })

    it('设置只接受合法的追寻值', () => {
        expect(validFocus('')).toBe(true)
        expect(validFocus('auto')).toBe(true)
        expect(validFocus('B03,B08')).toBe(true)
        expect(validFocus('B03,B03')).toBe(false)
        expect(validFocus('B09,B01')).toBe(false)
        expect(normalizeSettings({ focus: 'B01,B07' }).focus).toBe('B01,B07')
        expect(normalizeSettings({ focus: 'nonsense' }).focus).toBe('')
    })
})
