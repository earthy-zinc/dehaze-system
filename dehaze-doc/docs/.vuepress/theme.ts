import { Page, PageFrontmatter } from "vuepress";
import { hopeTheme } from "vuepress-theme-hope";
import navbar from "./navbar";
import sidebar from "./sidebar";

export default hopeTheme({
  hostname: "",
  author: {
    name: "earthy zinc",
    url: "https://gitee.com/earthy-zinc",
  },
  logo: "/tou.jpg",
  repo: "earthy-zinc/reading-note",
  repoLabel: "Gitee",
  repoDisplay: false,
  docsBranch: "master",
  docsDir: "dehaze-doc/docs",
  navbar,
  sidebar,
  encrypt: {
    config: {
      "/学术课程/学术经验/实用命令": "20230914",
      "/学术课程/研究日常/服务器教程": "20230914",
    },
  },
  markdown: {
    align: true,
    attrs: true,
    demo: true,
    figure: true,
    gfm: true,
    imgLazyload: true,
    imgSize: true,
    include: true,
    mark: true,
    sub: true,
    sup: true,
    tabs: true,

    echarts: true,
    markmap: true,
    mermaid: true,
    plantuml: true,

    codeTabs: true,
    preview: true,
  },
  plugins: {
    icon: {
      assets: "fontawesome-with-brands",
    },
    slimsearch: {
      indexContent: true,
      suggestion: true,
    },
  },
});
