// ============================================================================
// format.ts — 与 Python 格式化一致的小工具（{x:.1%}、{x:,.0f}）
// ============================================================================

/** 0.0123 → "1.2%"（digits 位小数） */
export function pct(x: number, digits = 0): string {
    return `${(x * 100).toFixed(digits)}%`
}

/** 12345.6 → "12,346"；无穷大显示为 "∞" */
export function int(x: number): string {
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞'
    return Math.round(x).toLocaleString('en-US')
}

/** 1.234 → "1.2" */
export function fixed(x: number, digits = 1): string {
    return x.toFixed(digits)
}
