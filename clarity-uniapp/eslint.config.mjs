import pluginVue from "eslint-plugin-vue";
import tsEslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import vueParser from "vue-eslint-parser";
import eslintConfigPrettier from "eslint-config-prettier";
import pluginPrettier from "eslint-plugin-prettier";
import globals from "globals";

export default [
  {
    ignores: [
      "dist/",
      "node_modules/",
      "public/",
      ".husky/",
      ".vscode/",
      ".idea/",
      "src/assets/",
      "*.config.*",
      "*.d.ts",
      "shime-uni.d.ts",
      "shims-uni.d.ts",
      "env.d.ts",
    ],
  },

  // Vue 3 推荐规则集（自动配置 vue-eslint-parser）
  ...pluginVue.configs["flat/recommended"],

  // TypeScript + Vue <script> 规则
  {
    files: ["**/*.ts", "**/*.vue"],
    languageOptions: {
      parser: vueParser,
      parserOptions: {
        parser: tsParser,
        ecmaVersion: "latest",
        sourceType: "module",
        extraFileExtensions: [".vue"],
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    plugins: {
      "@typescript-eslint": tsEslint,
      prettier: pluginPrettier,
    },
    rules: {
      // TS 推荐规则集
      ...tsEslint.configs.recommended.rules,

      // 项目自定义放宽
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-empty-function": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/no-require-imports": "off",

      // Vue 自定义放宽（recommended 已含基础规则）
      "vue/multi-word-component-names": "off",
      "vue/no-reserved-component-names": "off",
      "vue/require-default-prop": "off",
      "vue/require-explicit-emits": "off",
      "vue/attributes-order": "off",
      "vue/max-attributes-per-line": "off",
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

      // Prettier
      "prettier/prettier": "error",
    },
    settings: {
      "vue/setupCompilerMacros": true,
    },
  },

  eslintConfigPrettier,
];
