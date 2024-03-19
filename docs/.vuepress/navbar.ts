import {navbar} from "vuepress-theme-hope";

let host = "http://47.120.48.158";

export default navbar([
    "/前端开发",
    "/后端开发",
    "/学术课程",
    "/工作效率",
    "/艺术设计",
    "/项目文档",
    {
        text: "项目展示",
        children: [
            {
                text: "游戏开发",
                children: [
                    {
                        text: "一起来玩贪吃蛇",
                        link: host + ":82/"
                    }
                ]
            },
            {
                text: "学术应用",
                children: [
                    {
                        text: "图像去雾系统",
                        link: host + ":83/"
                    }
                ]
            },
            {
                text: "土味博客",
                children: [
                    {
                        text: "小沛の个人博客",
                        link: host + ":84/"
                    }
                ]
            },
            {
                text: "土味外卖",
                children: [
                    {
                        text: "餐厅后台管理系统",
                        link: host + ":85/"
                    },
                    {
                        text: "点单消费(手机访问)",
                        link: host + ":85/"
                    }
                ]
            },
            {
                text: "土味商城",
                children: [
                    {
                        text: "卖家订单管理系统",
                        link: host + ":86/"
                    },
                    {
                        text: "下单购物",
                        link: host + ":87/"
                    }
                ]
            },
        ]
    },
]);
