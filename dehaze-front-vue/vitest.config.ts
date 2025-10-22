import { defineConfig } from "vitest/config";
export default defineConfig({
  test: {
    // 测试环境
    environment: "jsdom",

    // 全局测试设置
    globals: true,

    // 包含的测试文件
    include: ["src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}"],

    // 排除的文件
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "**/cypress/**",
      "**/.{idea,git,cache,output,temp}/**",
      "**/e2e/**",
    ],

    // 覆盖率配置
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      include: ["src/**/*.{js,ts,vue}"],
      exclude: [
        "src/main.ts",
        "src/**/*.d.ts",
        "src/**/*.spec.ts",
        "src/**/*.test.ts",
        "src/typings/**",
        "src/assets/**",
        "src/**/__tests__/**",
        "src/**/__mocks__/**",
      ],
      // 覆盖率阈值
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },

    // 测试设置文件
    setupFiles: ["./src/__tests__/setup.ts", "./vitest.setup.ts"],

    // 依赖优化配置（用于 vitest-canvas-mock）
    deps: {
      optimizer: {
        web: {
          include: ["vitest-canvas-mock"],
        },
      },
    },

    // 测试超时时间
    testTimeout: 10000,

    // 钩子超时时间
    hookTimeout: 10000,

    // 启用 UI 界面
    ui: true,

    // 并发运行测试（Canvas mock 需要单线程）
    pool: "threads",

    // 环境选项
    environmentOptions: {
      jsdom: {
        resources: "usable",
      },
    },

    // 模拟选项
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,
  },
});
