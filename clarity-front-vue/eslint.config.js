// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";

import pluginVue from "eslint-plugin-vue";
import eslintConfigPrettier from "eslint-config-prettier";
import pluginPrettier from "eslint-plugin-prettier";
import tsEslint from "@typescript-eslint/eslint-plugin";
import vueParser from "vue-eslint-parser";
import fs from "fs";

const autoImportConfig = JSON.parse(
  fs.readFileSync("./.eslintrc-auto-import.json", "utf8")
);

export default [
  {
    ignores: [
      "dist/",
      "node_modules/",
      "public/",
      ".husky/",
      ".vscode/",
      ".idea/",
      "*.sh",
      "*.md",
      "src/assets/",
      ".eslintrc.cjs",
      ".prettierrc.cjs",
      ".stylelintrc.cjs",
      "**/*.js",
    ],
  },
  {
    files: ["**/*.ts", "**/*.js", "**/*.vue"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parser: vueParser,
      parserOptions: {
        parser: "@typescript-eslint/parser",
        project: "./tsconfig.json",
        extraFileExtensions: [".vue"],
      },
      globals: {
        ...autoImportConfig.globals,
        browser: true,
        es2021: true,
        node: true,
        OptionType: "readonly",
      },
    },
    plugins: {
      vue: pluginVue,
      "@typescript-eslint": tsEslint,
      prettier: pluginPrettier,
    },
    rules: {
      // Vue rules
      "vue/multi-word-component-names": "off",
      "vue/no-v-model-argument": "off",
      "vue/no-reserved-component-names": "off",
      "vue/custom-event-name-casing": "off",
      "vue/attributes-order": "off",
      "vue/one-component-per-file": "off",
      "vue/html-closing-bracket-newline": "off",
      "vue/max-attributes-per-line": "off",
      "vue/multiline-html-element-content-newline": "off",
      "vue/singleline-html-element-content-newline": "off",
      "vue/attribute-hyphenation": "off",
      "vue/require-default-prop": "off",
      "vue/require-explicit-emits": "off",
      "vue/html-self-closing": [
        "error",
        {
          html: {
            void: "always",
            normal: "never",
            component: "always",
          },
          svg: "always",
          math: "always",
        },
      ],

      // TypeScript rules
      "@typescript-eslint/no-empty-function": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/ban-ts-ignore": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/ban-types": "off",
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/no-var-requires": "off",
      "@typescript-eslint/no-use-before-define": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
      "@typescript-eslint/no-unused-vars": "off",

      // Prettier rules
      "prettier/prettier": [
        "error",
        {
          useTabs: false,
        },
      ],
    },
    settings: {
      "vue/setupCompilerMacros": true,
    },
  },
  eslintConfigPrettier,
  ...storybook.configs["flat/recommended"],
  {
    // ai-chat 组件层红线：零 store、零 SDK 运行时、零 vue-router、零 ElMessage。
    // 契约类型（Message/Block/Plan/Interrupt/Usage/Artifact/Feedback 等）允许 import type，
    // 组件不得引用 clarity-sdk-js 的任何运行时值（API/函数/常量）。
    files: ["src/components/ai-chat/**/*.{ts,vue}"],
    rules: {
      "@typescript-eslint/no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "vue-router",
              message: "ai-chat 组件层禁止路由依赖（红线）",
            },
            {
              name: "clarity-sdk-js",
              allowTypeImports: true,
              message:
                "ai-chat 组件层仅允许 import type 契约类型，禁止引用 SDK 运行时值（红线）",
            },
            {
              name: "element-plus",
              importNames: ["ElMessage", "ElMessageBox", "ElNotification"],
              message: "ai-chat 组件层禁止 UI 反馈（红线）",
            },
          ],
          patterns: [
            {
              group: ["@/store/**", "@/router/**"],
              message: "ai-chat 组件层禁止访问 store/router（红线）",
            },
          ],
        },
      ],
    },
  },
];
