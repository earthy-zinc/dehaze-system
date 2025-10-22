import pluginVue from "eslint-plugin-vue";
import eslintConfigPrettier from "eslint-config-prettier";
import pluginPrettier from "eslint-plugin-prettier";
import tsEslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
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
];
