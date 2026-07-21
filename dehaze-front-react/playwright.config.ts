import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E 测试配置
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  // 测试目录
  testDir: "./e2e",

  // 最大失败次数
  maxFailures: 2,

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
    // 基础 URL
    baseURL: "http://localhost:5173",

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

  // 配置不同的浏览器项目
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },

    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },

    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },

    // 移动端浏览器测试（可选）
    {
      name: "Mobile Chrome",
      use: { ...devices["Pixel 5"] },
    },
    {
      name: "Mobile Safari",
      use: { ...devices["iPhone 12"] },
    },
  ],

  // 在测试前启动开发服务器
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 12000,
  },
});
