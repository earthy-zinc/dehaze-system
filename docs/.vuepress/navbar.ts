import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "前端开发",
    prefix: "/前端开发/",
    children: [
      {
        text: "前端基础",
        link: "前端基础/CSS.md"
      },
      {
        text: "前端进阶",
        link: "前端进阶/Babel.md"
      },
      {
        text: "前端框架",
        link: "前端框架/Vue2.md"
      },
    ],
  },
  {
    text: "后端开发",
    prefix: "/后端开发/",
    children: [
      {
        text: "中间件",
        link: "中间件/Netty.md",
      },
      {
        text: "大数据",
        link: "大数据/Flink 练习.md",
      },
      {
        text: "开发知识",
        link: "开发知识/CDN.md",
      },
      {
        text: "操作系统",
        link: "操作系统/Linux.md",
      },
      {
        text: "数据库",
        link: "数据库/MySQL.md",
      },
      {
        text: "算法",
        link: "算法/LeetCode.md",
      },
      {
        text: "编程语言",
        link: "编程语言/Java基础.mc",
      },
      {
        text: "部署工具",
        link: "部署工具/Docker.md",
      },
    ]
  },
  {
    text: "学术课程",
    link: "/学术课程/学术论文/LaTeX公式.md",
  },
  {
    text: "工作效率",
    link: "/工作效率/Git.md",
  },
  {
    text: "项目文档",
    link: "/项目文档/API网关/项目概要.md",
  },


]);
