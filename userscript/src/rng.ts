// ============================================================================
// rng.ts — 随机数源。实战用 Math.random；测试用可复现的带种子生成器。
// ============================================================================

export interface Rng {
    /** [0, 1) 均匀分布 */
    random(): number
}

export const defaultRng: Rng = { random: Math.random }

/** mulberry32：32 位种子、周期 2^32，足够蒙特卡洛估算使用 */
export function seededRng(seed: number): Rng {
    let a = seed >>> 0
    return {
        random() {
            a = (a + 0x6d2b79f5) >>> 0
            let t = a
            t = Math.imul(t ^ (t >>> 15), t | 1)
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296
        }
    }
}
