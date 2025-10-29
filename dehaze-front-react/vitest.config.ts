import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";
import { playwright } from "@vitest/browser-playwright";
import storybookTest from "@storybook/addon-vitest/vitest-plugin";
import path from "path";

export default defineConfig((configEnv) =>
  mergeConfig(
    viteConfig(configEnv),
    defineConfig({
      test: {
        projects: [
          {
            extends: true,
            test: {
              name: "unit",
              environment: "jsdom",
              setupFiles: ["./vitest.setup.ts"],
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
                "**/*.mdx",
                "**/*.stories.@(js|jsx|mjs|ts|tsx)",
                "**/e2e/**",
              ],

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
          },
          {
            extends: true,
            plugins: [
              storybookTest({
                // The location of your Storybook config, main.js|ts
                configDir: path.join(__dirname, ".storybook"),
                // This should match your package.json script to run Storybook
                // The --ci flag will skip prompts and not open a browser
                storybookScript: "pnpm storybook --ci",
              }),
            ],
            test: {
              name: "storybook",
              // Enable browser mode
              browser: {
                enabled: true,
                headless: true,
                provider: playwright(),
                instances: [{ browser: "chromium" }],
              },
              // 排除的文件
              exclude: [
                "**/node_modules/**",
                "**/dist/**",
                "**/cypress/**",
                "**/.{idea,git,cache,output,temp}/**",
                "src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}",
                "**/e2e/**",
              ],
              setupFiles: ["./.storybook/vitest.setup.ts"],
            },
          },
        ],
        // 测试环境
        environment: "jsdom",
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
        // 启用 UI 界面
        ui: false,
      },
    })
  )
);
