#!/usr/bin/env node
// 用法：node scripts/hooks/run-ruff.mjs <ruff 可执行文件> <文件...>
// 为什么需要这层：lint-staged 会把同一组的多条命令并发执行，而 `ruff check --fix` 与
// `ruff format` 操作同一批文件，并发会互相覆盖；`&&` 又不能写在命令里（lint-staged 不经 shell）。
// 所以两步放进同一个进程串行跑。
import { spawnSync } from 'node:child_process'

const [ruff, ...files] = process.argv.slice(2)

const run = (args) => {
  const r = spawnSync(ruff, args, { stdio: 'inherit' })
  if (r.error) {
    console.error(`run-ruff: 无法执行 ${ruff}: ${r.error.message}`)
    process.exit(1)
  }
  if (r.status !== 0) process.exit(r.status ?? 1)
}

run(['check', '--fix', ...files])
run(['format', ...files])
