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

    // 设置文件
    setupFiles: ["./vitest.setup.ts"],

    // 超时配置
    testTimeout: 60000,
    hookTimeout: 60000,

    // Mock 配置
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,

    // 覆盖率配置（集成测试不生成代码覆盖率）
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      include: ["src/**/*.{js,ts}"],
      exclude: ["node_modules/", "dist/", "test/", "src/**/*.d.ts", "src/types/**"],
      // 不设置阈值（集成测试仅验证 API 可用性，不覆盖源代码）
    },

    // 全局并发
    maxConcurrency: 10,

    // 所有测试文件并行运行
    fileParallelism: true,
  },
});
