import {navbar} from "vuepress-theme-hope";

export default navbar([
    "/前端开发",
    "/后端开发",
    "/学术课程",
    "/通用工具",
    "/项目文档",
    {
        text: "项目展示",
        children: [
            {
                text: "游戏开发",
                children: [
                    {
                        text: "一起来玩贪吃蛇",
                        // 90
                        link: "/kob"
                    }
                ]
            },
            {
                text: "学术应用",
                children: [
                    {
                        text: "图像去雾系统",
                        // 91,92
                        link: "/dehaze-front"
                    }
                ]
            },
            {
                text: "土味博客",
                children: [
                    {
                        text: "小沛博客v1",
                        // 93
                        link: "/pei-blog"
                    },
                    {
                        text: "小沛博客v2",
                        link: "/wpx-blog"
                    },
                    {
                        text: "小沛博客管理后台",
                        link: "/wpx-blog-admin"
                    }
                ]
            },
            {
                text: "土味外卖",
                children: [
                    {
                        text: "餐厅后台管理系统",
                        link: "/earthy-delivery/backend"
                    },
                    {
                        text: "点单消费(手机访问)",
                        link: "/earthy-delivery/front"
                    }
                ]
            },
            {
                text: "土味商城",
                children: [
                    {
                        text: "卖家订单管理系统",
                        // 94
                        link: "/tuwei-mall/backend"
                    },
                    {
                        text: "下单购物",
                        link: "/tuwei-mall/front"
                    }
                ]
            },
        ]
    },
]);
