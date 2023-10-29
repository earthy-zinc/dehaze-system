import {navbar} from "vuepress-theme-hope";

export default navbar([
    "/前端开发",
    "/后端开发",
    "/学术课程",
    "/工作效率",
    "/项目文档",
    {
        text: "项目展示",
        children: [
            {
                text: "游戏开发",
                children: [
                    {
                        text: "贪吃蛇",
                        link: "http://10.16.36.157/greedy_snake"
                    }
                ]
            },
            {
                text: "学术应用",
                children: [
                    {
                        text: "去雾系统",
                        link: "/error"
                    }
                ]
            },
            {
                text: "土味博客",
                children: [
                    {
                        text: "小沛博客",
                        link: "/error"
                    }
                ]
            },
            {
                text: "土味外卖",
                children: [
                    {
                        text: "餐厅管理",
                        link: "http://10.16.36.157:8081/backend/index.html"
                    },
                    {
                        text: "点单消费(手机访问)",
                        link: "http://10.16.36.157:8081/front/index.html"
                    }
                ]
            },
            {
                text: "土味商城",
                children: [
                    {
                        text: "卖家管理",
                        link: "/error"
                    },
                    {
                        text: "下单购物",
                        link: "/error"
                    }
                ]
            }
        ]
    }
]);
