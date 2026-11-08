module.exports = {
  root: true,
  extends: ["taro/react", "plugin:prettier/recommended"],
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: "latest",
    sourceType: "module",
  },
  env: {
    browser: true,
    node: true,
    es6: true,
  },
  settings: {
    react: {
      version: "detect",
    },
  },
  rules: {
    "react/jsx-uses-react": "off",
    "react/react-in-jsx-scope": "off",
    // TypeScript rules
    "@typescript-eslint/no-empty-function": "off",
    "@typescript-eslint/no-explicit-any": "off",
    "@typescript-eslint/no-non-null-assertion": "off",
    "@typescript-eslint/ban-ts-comment": "off",
    "@typescript-eslint/no-var-requires": "off",
    "@typescript-eslint/no-use-before-define": "off",
    "@typescript-eslint/explicit-module-boundary-types": "off",
    "@typescript-eslint/no-unused-vars": [
      "warn",
      { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
    ],
    // Prettier
    "prettier/prettier": "warn",
  },
  ignorePatterns: [
    "dist/",
    "node_modules/",
    "public/",
    ".husky/",
    ".vscode/",
    ".idea/",
    "*.sh",
    "*.md",
    "src/assets/",
    "config/",
    "babel.config.js",
    ".eslintrc.cjs",
    ".prettierrc.cjs",
    ".stylelintrc.cjs",
    "jest.config.js",
    // RN 构建的配置与 postcss 插件由 Metro 直接以 CommonJS 加载，不走 babel
    "metro.config.js",
    "index.js",
    "plugins/",
  ],
};
