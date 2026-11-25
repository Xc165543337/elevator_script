// ============================================================================
// main.ts — 油猴脚本入口：在游戏页面上挂载面板（同一页面只挂一次）
// ============================================================================

import { mountPanel } from './ui.ts'

const FLAG = '__elevatorFarmerMounted'
const g = globalThis as unknown as Record<string, unknown>

if (!g[FLAG]) {
    g[FLAG] = true
    if (document.body) mountPanel()
    else document.addEventListener('DOMContentLoaded', () => mountPanel(), { once: true })
}
