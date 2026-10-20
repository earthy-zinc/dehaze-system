/**
 * 跑 SDK 套件前的残留 runner 拦截（由 package.json 的 pre* 钩子在 vitest 启动前调用）。
 *
 * 为什么需要：`timeout`/工具超时只杀掉 pnpm·vitest 父进程，其 worker 仍会继续跑；残留进程与
 * 新进程抢同一账号的 session（deviceType 不参与判定，登录即按在线总数踢最早），表现为成片
 * A0230「会话失效」，是 SDK 集成测试最耗时的排查项。故在 vitest 启动前先探一次，发现残留立刻
 * 非零退出，而不是跑完几百条用例再人工定位。
 *
 * 为什么钩子在 vitest 之外：放进 vitest.globalSetup 后 `pgrep -f vitest.mjs` 会先匹配到当前
 * vitest 进程自身，恒为假阳性。
 *
 * 文件名刻意不含 `vitest.mjs`：脚本自身 cmdline 会被 `pgrep -af vitest.mjs` 匹配到（子串命中），
 * 故这里额外按自身 pid 过滤以容忍改名。
 */
import { execFileSync } from "node:child_process";

let stdout = "";
try {
  stdout = execFileSync("pgrep", ["-af", "vitest.mjs"], { encoding: "utf8" });
} catch {
  // pgrep 退出码 1 = 无匹配；无 pgrep 的平台（Windows）同样按"无残留"放行
  process.exit(0);
}

const stale = stdout
  .split("\n")
  .filter((line) => line.trim() !== "")
  .filter((line) => Number(line.split(/\s+/, 1)[0]) !== process.pid);

if (stale.length === 0) {
  process.exit(0);
}

console.error(
  "检测到残留 vitest 进程，已中止本次运行——它会与新进程抢同一账号 session（成片 A0230）："
);
console.error(stale.join("\n"));
console.error("清理命令：pkill -f vitest.mjs");
process.exit(1);
