import {Page, PageFrontmatter} from "vuepress";
import { hopeTheme } from "vuepress-theme-hope";
import navbar from "./navbar";
import sidebar from "./sidebar";

export default hopeTheme({
  hostname: "http://192.168.31.2",
  author: {
    name: "earthy zinc",
    url: "https://gitee.com/earthy-zinc",
  },
  iconAssets: "fontawesome-with-brands",
  logo: "/tou.jpg",
  repo: "earthy-zinc/reading-note",
  repoLabel: "Gitee",
  repoDisplay: false,
  docsDir: "docs",
  navbar,
  sidebar,
  encrypt: {
    config: {
      "/学术课程/学术经验/" : "142536",
      "/学术课程/学术经验/实用命令.html" : "20230914",
      "/学术课程/研究日常/": "20230914",
    },
  },
  plugins: {
    autoCatalog: {
      index: true,
      titleGetter: (page) => "  " + page.title,
      orderGetter: (page: Page) => page.frontmatter.order as number || -9999
    },
    mdEnhance: {
      align: true,
      attrs: true,
      card: true,
      codetabs: true,
      demo: true,
      figure: true,
      gfm: true,
      imgLazyload: true,
      imgSize: true,
      include: true,
      mathjax: true,
      mark: true,
      sub: true,
      sup: true,
      tabs: true,
    },
  },
});
