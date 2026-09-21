/**
 * 开关依赖型端到端用例的独立配置（默认套件 vitest.config.ts 只收 `test/modules/**`）。
 *
 * 用途：`USE_MULTI_POINT` 等由后端配置开关决定行为的用例，不能在默认套件里跑
 * （dev 默认 false，全量基线建立在"多端共存"语义上），必须单独窗口执行。
 *
 *   cd dehaze-sdk-js && SESSION_MULTI_POINT=true pnpm test:e2e
 */
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
    include: ["test/e2e/**/*.test.ts"],
    includeTaskLocation: true,
    fileParallelism: false,
    setupFiles: ["./test/config/vitest.setup.ts"],
    globalSetup: "./test/config/vitest.globalSetup.ts",
    testTimeout: 120000,
    hookTimeout: 120000,
    reporters: [compactReporter],
  },
});
