import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "#": path.resolve(__dirname, "./test"),
    },
  },
  test: {
    // 全局函数支持
    globals: true,

    // Node.js 环境
    environment: "node",

    // 包含的测试文件
    include: ["test/**/*.test.ts"],

    // 设置文件（每个测试文件执行前运行，仅做必要的环境初始化）
    setupFiles: ["./test/vitest.setup.ts"],

    // 全局设置（整个测试运行只执行一次，用于一次性清理 Redis 缓存）
    // 与 setupFiles 配合使用：globalSetup 清理一次，setupFiles 不再重复清理
    globalSetup: "./test/vitest.globalSetup.ts",

    // 超时配置
    testTimeout: 120000,
    hookTimeout: 120000,

    // 覆盖率配置（集成测试不生成代码覆盖率）
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      include: ["src/**/*.{js,ts}"],
      exclude: ["node_modules/", "dist/", "test/", "src/**/*.d.ts", "src/types/**"],
    },

    // 集成测试共享同一后端实例（Redis/MySQL），并行运行会导致缓存污染和数据竞争
    // 必须串行执行以保证测试隔离性和可重复性
    fileParallelism: false,
  },
});
