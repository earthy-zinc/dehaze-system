module.exports = {
  extends: [
    "stylelint-config-standard",
    "stylelint-config-recommended-scss",
    "stylelint-config-recess-order",
  ],
  overrides: [
    {
      files: ["**/*.less"],
      customSyntax: "postcss-less",
    },
    {
      files: ["**/*.scss"],
      customSyntax: "postcss-scss",
    },
  ],
  rules: {
    "import-notation": "string",
    "selector-class-pattern": null,
    "custom-property-pattern": null,
    "keyframes-name-pattern": null,
    "no-descending-specificity": null,
    "no-empty-source": null,
    // 允许 Taro/小程序的 rpx 单位
    "unit-no-unknown": [
      true,
      {
        ignoreUnits: ["rpx"],
      },
    ],
    // 允许小程序内置类型选择器（page、mp 等）
    "selector-type-no-unknown": [
      true,
      {
        ignoreTypes: ["page", "mp"],
      },
    ],
    // 允许未知伪类（小程序组件伪类等）
    "selector-pseudo-class-no-unknown": [
      true,
      {
        ignorePseudoClasses: ["global", "export", "deep"],
      },
    ],
    // 允许未知 at-rules（Less/SCSS 语法）
    "at-rule-no-unknown": [
      true,
      {
        ignoreAtRules: ["apply", "use", "import", "reference", "plugin"],
      },
    ],
    // Less/SCSS 嵌套会产生重复选择器误报
    "no-duplicate-selectors": null,
  },
};
