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
                        link: "/greedy_snake"
                    }
                ]
            },
            {
                text: "学术应用",
                children: [
                    {
                        text: "去雾系统",
                        link: "/dehaze_tool"
                    }
                ]
            },
            {
                text: "土味博客",
                children: [
                    {
                        text: "小沛博客",
                        link: "/blog"
                    }
                ]
            },
            {
                text: "土味外卖",
                children: [
                    {
                        text: "餐厅管理",
                        link: "/deliver_shoper"
                    },
                    {
                        text: "点单消费",
                        link: "/deliver_customer"
                    }
                ]
            },
            {
                text: "土味商城",
                children: [
                    {
                        text: "卖家管理",
                        link: "/shop_shoper"
                    },
                    {
                        text: "下单购物",
                        link: "/shop_customer"
                    }
                ]
            }
        ]
    }
]);
