import { defineConfig } from "vitest/config";
import path from "path";
import compactReporter from "./test/config/compact-reporter";

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
    include: ["test/modules/**/*.test.ts"],
    // 供 reporter 输出失败用例的 文件:行号
    includeTaskLocation: true,
    // 串行运行测试文件，避免并行测试文件共享数据库时互相删除数据
    fileParallelism: false,
    setupFiles: ["./test/config/vitest.setup.ts"],
    globalSetup: "./test/config/vitest.globalSetup.ts",
    testTimeout: 120000,
    hookTimeout: 120000,
    // 自定义紧凑报告器：brief.json（简要，供大模型查阅）+ detail.json（NDJSON 全量明细，供疑难排查）
    reporters: [compactReporter],
  },
});
