import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "前端开发",
    prefix: "/前端开发/",
    children: ["前端基础","前端进阶","前端框架"],
  },
  {
    text: "后端开发",
    prefix: "/后端开发/",
    children: ["中间件","大数据","开发知识","操作系统","数据库","算法","编程语言","部署工具"]
  },
  {
    text: "学术课程",
    prefix: "/学术课程/",
    children: ["学术经验","学术论文","深度学习","研究生课程","研究生面试"]
  },
  {
    text: "工作效率",
    link: "/工作效率/",
  },
  {
    text: "项目文档",
    prefix: "/项目文档/",
    children: ["API网关","中间件设计","土味商城","抽奖系统","沛信","贪吃蛇"],
  },
]);
