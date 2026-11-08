// babel-preset-taro 更多选项和默认值：
// https://github.com/NervJS/taro/blob/next/packages/babel-preset-taro/README.md
//
// RN 构建不注入 Taroify 样式：其 scss 用了 Dart Sass 的 @forward 模块语法，
// Taro 的 RN sass 编译链不解析该语法的相对路径，且 RN 下这套 DOM 样式本就不可用，
// 由 src/platform/ui 的 RN 组件实现替代。
const isRN = process.env.TARO_ENV === "rn";

module.exports = {
  plugins: [
    [
      "import",
      {
        libraryName: "@taroify/core",
        libraryDirectory: "",
        style: !isRN,
      },
      "@taroify/core",
    ],
    [
      "import",
      {
        libraryName: "@taroify/icons",
        libraryDirectory: "",
        camel2DashComponentName: false,
        style: isRN ? false : () => "@taroify/icons/style",
        customName: (name) =>
          name === "Icon"
            ? "@taroify/icons/van/VanIcon"
            : `@taroify/icons/${name}`,
      },
      "@taroify/icons",
    ],
  ],
  presets: [
    [
      "taro",
      {
        framework: "react",
        ts: true,
        compiler: "webpack5",
      },
    ],
  ],
};
