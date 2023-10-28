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

  docsDir: "docs",

  // navbar
  navbar,

  // sidebar
  sidebar,

  plugins: {
    autoCatalog: true,
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
