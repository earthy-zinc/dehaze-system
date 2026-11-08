#!/usr/bin/env node
// 在指定包目录内执行 lint 链。为什么要这个脚本而不是在 lint-staged 配置里写 `cd x && eslint`：
// lint-staged 用 execaCommand 直接 spawn，不经过 shell，`cd x && ...` 会被当成可执行文件名报 ENOENT。
// 用法：node scripts/hooks/run-lint.mjs <包目录> <绝对文件路径...>
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { FRONTEND, extOf } from './lint-plan.mjs'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const [pkgDir, ...files] = process.argv.slice(2)

const pkg = FRONTEND.find((p) => p.dir === pkgDir)
if (!pkg) {
  console.error(`run-lint: 未知的包目录 ${pkgDir}`)
  process.exit(1)
}

const cwd = path.join(root, pkgDir)
const isWin = process.platform === 'win32'

// Windows 下需要 shell 才能解析 node_modules/.bin 里的 .cmd 包装；参数含空格时必须自己加引号。
const winArg = (a) => (/[\s"]/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a)

const run = (tool, args) => {
  const bin = path.join(cwd, 'node_modules', '.bin', tool)
  const finalArgs = isWin ? args.map(winArg) : args
  const r = spawnSync(bin, finalArgs, { cwd, stdio: 'inherit', shell: isWin })
  if (r.error) {
    console.error(`run-lint: 无法执行 ${tool}: ${r.error.message}`)
    process.exit(1)
  }
  if (r.status !== 0) process.exit(r.status ?? 1)
}

const rel = (f) => path.relative(cwd, f)

for (const [tool, exts] of [
  ['eslint', pkg.eslint],
  ['stylelint', pkg.stylelint],
  ['prettier', pkg.prettier],
]) {
  if (!exts) continue
  const matched = files.filter((f) => exts.includes(extOf(f)))
  if (matched.length === 0) continue

  if (tool === 'eslint') run('eslint', ['--fix', ...matched.map(rel)])
  else if (tool === 'stylelint') run('stylelint', ['--fix', ...matched.map(rel)])
  else run('prettier', ['--write', ...matched.map(rel)])
}
