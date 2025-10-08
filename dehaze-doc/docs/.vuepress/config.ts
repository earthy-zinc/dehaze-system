import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { cachePlugin } from "@vuepress/plugin-cache";

import theme from "./theme";

export default defineUserConfig({
  base: "/",
  lang: "zh-CN",
  title: "土味锌的阅读笔记",
  description: "全栈开发学习笔记",
  bundler: viteBundler(),
  theme,
  plugins: [
    cachePlugin({
      // 配置项
    }),
  ],
});
