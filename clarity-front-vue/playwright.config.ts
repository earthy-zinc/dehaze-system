import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";
import dotenv from "dotenv";

/**
 * 预览端口取自 .env.test（与 vite.config.ts 同源），避免端口写两处而不一致。
 * 按配置文件所在目录定位而非 cwd：IDE 的 Playwright 扩展等从仓库根发起调用时 cwd 不是本项目目录。
 */
const testEnv = dotenv.parse(
  fs.readFileSync(
    path.join(path.dirname(fileURLToPath(import.meta.url)), ".env.test"),
    "utf-8"
  )
);
const previewPort = testEnv.VITE_PREVIEW_PORT;
if (!previewPort) {
  throw new Error("缺少 VITE_PREVIEW_PORT：请确认 clarity-front-vue/.env.test");
}

/**
 * 对已部署环境（如 pre）跑同一套用例时用 E2E_BASE_URL 覆盖：此时不再本地起 preview，
 * 直接打该环境的冻结产物。验证码所在 Redis 库须随环境切换（见 e2e/utils/auth.ts 的
 * E2E_REDIS_DATABASE），否则会读到本机环境的库、登录必然失败。
 */
const externalBaseURL = process.env.E2E_BASE_URL;
const baseURL = externalBaseURL ?? `http://localhost:${previewPort}`;

/**
 * Playwright E2E 测试配置
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./e2e",

  // 产物统一落 test_result/：默认 outputDir（test-results/）会与 reporter 输出分裂成两个结果目录
  outputDir: "test_result/artifacts",

  // 完全并行运行测试
  fullyParallel: true,

  // CI 环境下禁止 retries
  forbidOnly: !!process.env.CI,

  // 失败重试次数
  retries: process.env.CI ? 2 : 0,

  // 并发 worker 数量
  workers: process.env.CI ? 1 : undefined,

  // 测试报告
  reporter: [
    ["html", { outputFolder: "test_result/playwright" }],
    ["json", { outputFile: "test_result/playwright/results.json" }],
    ["list"],
  ],

  // 所有测试的共享配置
  use: {
    // 基础 URL（指向冻结预览产物，见下方 webServer）
    baseURL,

    // 失败时截图
    screenshot: "only-on-failure",

    // 失败时录制视频
    video: "retain-on-failure",

    // 首次重试时收集追踪
    trace: "on-first-retry",

    // 默认超时时间
    actionTimeout: 15000,
    navigationTimeout: 30000,
  },

  /**
   * 默认只跑 chromium：页面级验证的目标是"页面能不能用"，跨浏览器矩阵是独立诉求，
   * 默认铺开会让每次回归的成本乘以浏览器数量。需要时用 `--project` 显式指定。
   */
  projects: [
    {
      name: "setup",
      testMatch: /auth\.setup\.ts/,
    },
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        storageState: "test_result/.auth/user.json",
      },
      dependencies: ["setup"],
    },
  ],

  /**
   * 页面测试跑构建产物（vite preview）而不是 dev server：dev server 会在源码或 SDK
   * 产物变动时整页 reload，把用例打断在任意中间状态上。
   * reuseExistingServer 必须为 false——复用到的可能是别人正在编辑源码时挂着的 dev server，
   * 那样测的既不是冻结产物，结果也不可复现。产物由 `pnpm test:e2e` 前置构建。
   */
  webServer: externalBaseURL
    ? undefined
    : {
        command: "pnpm preview",
        url: baseURL,
        reuseExistingServer: false,
        timeout: 60000,
      },
});
