import { defineConfig } from "vitest/config";
import path from "path";
import compactReporter from "./test/compact-reporter";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "#": path.resolve(__dirname, "./test"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["test/**/*.test.ts"],
    // 供 reporter 输出失败用例的 文件:行号
    includeTaskLocation: true,
    setupFiles: ["./test/vitest.setup.ts"],
    globalSetup: "./test/vitest.globalSetup.ts",
    testTimeout: 120000,
    hookTimeout: 120000,
    // 紧凑 JSON 输出（单行，仅汇总 + 失败详情），替代默认排版化报告
    reporters: [compactReporter],
  },
});
