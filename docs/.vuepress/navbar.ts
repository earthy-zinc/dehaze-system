import {navbar} from "vuepress-theme-hope";

let host = "http://10.16.90.26";

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
                        text: "一起来玩贪吃蛇",
                        link: host + "/greedy_snake/"
                    }
                ]
            },
            {
                text: "学术应用",
                children: [
                    {
                        text: "图像去雾系统",
                        link: "http://10.16.90.26/dehaze_front/#/component/demo/dehaze"
                    }
                ]
            },
            {
                text: "土味博客",
                children: [
                    {
                        text: "小沛の个人博客",
                        link: host + "/pei_blog"
                    }
                ]
            },
            {
                text: "土味外卖",
                children: [
                    {
                        text: "餐厅后台管理系统",
                        link: host + "/deliver_manager"
                    },
                    {
                        text: "点单消费(手机访问)",
                        link: host + "/deliver_customer"
                    }
                ]
            },
            {
                text: "土味商城",
                children: [
                    {
                        text: "卖家订单管理系统",
                        link: host + "/youlai_mall"
                    },
                    {
                        text: "下单购物",
                        link: host + "/youlai_app"
                    }
                ]
            },
            {
                text: "其他",
                children: [
                    {
                        text: '资源下载',
                        link: host + '/pei_drive'
                    },
                    {
                        text: '在线编程',
                        link: host + '/code_ide'
                    },
                ]
            },
        ]
    },
]);
