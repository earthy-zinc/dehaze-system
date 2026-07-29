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
    globals: true,
    environment: "node",
    include: ["test/**/*.test.ts"],
    setupFiles: ["./test/vitest.setup.ts"],
    globalSetup: "./test/vitest.globalSetup.ts",
    testTimeout: 120000,
    hookTimeout: 120000,
  },
});
