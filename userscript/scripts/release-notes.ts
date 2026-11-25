// ============================================================================
// release-notes.ts — 发布前校验版本标签，并从 CHANGELOG.md 提取该版本的说明
//
// 用法: node userscript/scripts/release-notes.ts <标签，如 v1.0.0> [输出文件]
//   - 标签必须与 userscript/package.json 的版本一致（v + version）
//   - userscript/CHANGELOG.md 中必须有该版本的条目（## [x.y.z]）
// 任一条件不满足即以非零状态退出，阻止发布。由 .github/workflows/release.yml 调用。
// ============================================================================

import { readFileSync, writeFileSync } from 'node:fs'

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** 标签是否对应该版本（v1.0.0 ↔ 1.0.0） */
export function tagMatchesVersion(tag: string, version: string): boolean {
    return tag === `v${version}`
}

/** 提取 CHANGELOG 中某版本的正文（不含版本标题）；缺失或为空时返回 null */
export function extractReleaseNotes(changelog: string, version: string): string | null {
    const lines = changelog.split(/\r?\n/)
    const header = new RegExp(`^## \\[${escapeRegExp(version)}\\]`)
    const start = lines.findIndex(line => header.test(line))
    if (start < 0) return null
    const next = lines.findIndex((line, index) => index > start && line.startsWith('## ['))
    const body = lines
        .slice(start + 1, next < 0 ? undefined : next)
        .join('\n')
        .trim()
    return body === '' ? null : body
}

function fail(message: string): never {
    console.error(message)
    process.exit(1)
}

if (import.meta.main) {
    const [tag, outFile] = process.argv.slice(2)
    if (!tag) fail('用法: node userscript/scripts/release-notes.ts <标签，如 v1.0.0> [输出文件]')

    const root = new URL('../', import.meta.url)
    const { version } = JSON.parse(readFileSync(new URL('package.json', root), 'utf-8')) as { version: string }
    if (!tagMatchesVersion(tag, version)) {
        fail(`标签 ${tag} 与 userscript/package.json 的版本 ${version} 不一致：请先更新版本，或使用标签 v${version}`)
    }

    const notes = extractReleaseNotes(readFileSync(new URL('CHANGELOG.md', root), 'utf-8'), version)
    if (!notes) fail(`userscript/CHANGELOG.md 中没有 ${version} 的内容：请先添加「## [${version}]」条目`)

    if (outFile) writeFileSync(outFile, `${notes}\n`, 'utf-8')
    else console.log(notes)
}
