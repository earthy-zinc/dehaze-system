// 仓库级 lint-staged 配置，由 .husky/pre-commit 在仓库根调用。
//
// 三条设计约束：
// 1. 前端命令交给 scripts/hooks/run-lint.mjs，由它在包目录内执行 —— 各包的 ESLint
//    （flat config / .eslintrc）、stylelint、prettier 配置都在包内，在仓库根执行会向上找不到配置。
//    不能在配置里写 `cd x && eslint ...`：lint-staged 用 execaCommand 直接 spawn，不经 shell，会 ENOENT。
// 2. 只处理暂存文件，stash 保护与「修复后自动 re-add」交给 lint-staged，不自造轮子。
// 3. 后端只放「单文件毫秒级」检查：Python 用 ruff，Go 用 gofmt。
//    Java 的 spotless:check 全项目约 21s 且要求存量全绿，不进 pre-commit（走 quality-gate 轮末格式化）。
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { planFor } from './scripts/hooks/lint-plan.mjs'

const ROOT = path.dirname(fileURLToPath(import.meta.url))
const RUN_LINT = path.join(ROOT, 'scripts', 'hooks', 'run-lint.mjs')
const RUN_RUFF = path.join(ROOT, 'scripts', 'hooks', 'run-ruff.mjs')
const RUFF = path.join(
  ROOT,
  'clarity-python',
  '.venv',
  process.platform === 'win32' ? 'Scripts' : 'bin',
  process.platform === 'win32' ? 'ruff.exe' : 'ruff'
)

const quote = (p) => JSON.stringify(p)

export default (files) => {
  const commands = []

  for (const plan of planFor(files, ROOT)) {
    const owned = [...new Set(plan.steps.flatMap((s) => s.files))]
    commands.push(`node ${quote(RUN_LINT)} ${quote(plan.cwd)} ${owned.map(quote).join(' ')}`)
  }

  // Python：ruff 按「离文件最近的 pyproject.toml / ruff.toml」解析配置，
  // 因此一次传入跨项目的文件也能各自命中正确规则；exclude 靠配置里的 force-exclude 生效。
  // 走 run-ruff.mjs 是为了让 check --fix 与 format 串行（lint-staged 的多条命令是并发的）。
  const pyFiles = files.filter((f) => f.endsWith('.py'))
  if (pyFiles.length > 0) {
    commands.push(`node ${quote(RUN_RUFF)} ${quote(RUFF)} ${pyFiles.map(quote).join(' ')}`)
  }

  // Go：只做格式化，go vet / 编译留给 pre-push 或人工
  const goFiles = files.filter((f) => f.endsWith('.go'))
  if (goFiles.length > 0) commands.push(`gofmt -w ${goFiles.map(quote).join(' ')}`)

  return commands
}
