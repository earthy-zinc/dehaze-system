import { defineUserConfig } from "vuepress";
import {autoCatalogPlugin} from "vuepress-plugin-auto-catalog";
// import fullTextSearchPlugin from "vuepress-plugin-full-text-search2";
import theme from "./theme";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "土味锌的阅读笔记",
  description: "全栈开发学习笔记",

  theme,
  // plugins: [
  //   autoCatalogPlugin({
  //     //插件选项
  //   }),
  //   // fullTextSearchPlugin
  // ],
});
