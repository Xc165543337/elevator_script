// ============================================================================
// build.mjs — 打包为单个油猴脚本 dist/elevator-farmer.user.js
//
// 版本号以 package.json 为唯一来源，写入脚本头与面板标题。
// 在 GitHub Actions 中（GITHUB_REPOSITORY 已设置）写入 @homepageURL；仅当 USERSCRIPT_AUTO_UPDATE=true
// （仓库公开，由 ci.yml 设置）时再写入 @updateURL / @downloadURL，指向最新 Release 的附件——
// 私有仓库的附件需要登录才能下载，油猴无法据此更新，写入反而会让更新检查一直失败。
// @namespace 固定不变：油猴按「名称 + 命名空间」识别脚本，本地构建与 CI 构建应视为同一脚本。
// 不压缩代码：便于用户审阅。
// ============================================================================

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { build } from 'esbuild'

const root = resolve(import.meta.dirname, '..')
const { version, description } = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf-8'))
export const FILE_NAME = 'elevator-farmer.user.js'

const repo = process.env.GITHUB_REPOSITORY || ''
const autoUpdate = Boolean(repo) && process.env.USERSCRIPT_AUTO_UPDATE === 'true'
const meta = [
    ['name', '无尽电梯 · 自动刷首领'],
    ['namespace', 'elevator-farmer'],
    ['version', version],
    ['description', description],
    ['match', 'https://elevator.slime.icu/elevator*'],
    ['grant', 'none'],
    ['run-at', 'document-idle'],
    ['noframes', '']
]
if (repo) meta.push(['homepageURL', `https://github.com/${repo}`])
if (autoUpdate) {
    const asset = `https://github.com/${repo}/releases/latest/download/${FILE_NAME}`
    meta.push(['updateURL', asset], ['downloadURL', asset])
}
const width = Math.max(...meta.map(([k]) => k.length)) + 2
const header = [
    '// ==UserScript==',
    ...meta.map(([k, v]) => `// @${k.padEnd(width - 1)}${v}`.trimEnd()),
    '// ==/UserScript=='
].join('\n')

const result = await build({
    entryPoints: [resolve(root, 'src/main.ts')],
    bundle: true,
    format: 'iife',
    target: 'es2022',
    charset: 'utf8',
    legalComments: 'none',
    define: { __APP_VERSION__: JSON.stringify(version) },
    write: false
})

mkdirSync(resolve(root, 'dist'), { recursive: true })
const out = resolve(root, 'dist', FILE_NAME)
writeFileSync(out, `${header}\n\n${result.outputFiles[0].text}`, 'utf-8')
console.log(
    `dist/${FILE_NAME}：v${version}，${(readFileSync(out).length / 1000).toFixed(0)} kB${autoUpdate ? `，自动更新：${repo} 最新 Release` : '，无自动更新'}`
)
