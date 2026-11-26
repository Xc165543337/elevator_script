// ============================================================================
// solver.ts — 价值表的求解方式
//
// 游戏页面的内容安全策略（default-src 'self'）禁止 blob: / data: Worker，油猴脚本又只有一个文件，
// 无法使用 Web Worker。因此在页面主线程分片求解：每片最多 sliceMs 毫秒，之后经 MessageChannel 让出，
// 让对局请求与界面照常进行（MessageChannel 不像定时器那样在后台标签页被限速）。
// ============================================================================

import { solve, solveSteps } from './mdp.ts'
import type { MdpConfig, Progress, ValueTables } from './mdp.ts'
import type { Build } from './model.ts'

/** 价值表的求解方式 */
export interface TableSolver {
    solve(build: Build, mapId: string, config: MdpConfig, progress?: Progress): Promise<ValueTables>
}

/** 一次性求解（测试用；在浏览器里会阻塞页面数分钟） */
export const blockingSolver: TableSolver = {
    solve: async (build, mapId, config, progress) => solve(build, mapId, config, progress)
}

/** 让出主线程：处理完已排队的事件（网络回调、界面）后继续 */
export function yieldToEventLoop(): Promise<void> {
    return new Promise(resolve => {
        const channel = new MessageChannel()
        channel.port1.addEventListener('message', () => {
            channel.port1.close()
            resolve()
        })
        channel.port1.start()
        channel.port2.postMessage(null)
    })
}

/** 分片求解：每片最多 sliceMs 毫秒（单层的首领求解可能略超） */
export function slicedSolver(sliceMs = 25): TableSolver {
    return {
        async solve(build, mapId, config, progress) {
            const steps = solveSteps(build, mapId, config, progress)
            let start = performance.now()
            for (;;) {
                const next = steps.next()
                if (next.done) return next.value
                if (performance.now() - start >= sliceMs) {
                    await yieldToEventLoop()
                    start = performance.now()
                }
            }
        }
    }
}
