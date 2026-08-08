import { defineConfig } from "vitest/config";
import path from "path";
import compactReporter from "./test/config/compact-reporter";

/** 纯前端单元测试，不加载集成测试的登录、Redis/MySQL 初始化。 */
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "#": path.resolve(__dirname, "./test"),
    },
  },
  test: {
    include: ["test/unit/**/*.test.ts"],
    includeTaskLocation: true,
    reporters: [compactReporter],
  },
});
