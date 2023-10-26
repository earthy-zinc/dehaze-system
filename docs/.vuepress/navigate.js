let frontEndNav = [
    {
        text: '前端基础', items: [
            {text: 'CSS', link: '/前端开发/前端基础/CSS.md'},
            {text: 'HTML', link: '/前端开发/前端基础/HTML.md'},
            {text: 'HTTP', link: '/前端开发/前端基础/HTTP.md'},
            {text: 'JavaScript', link: '/前端开发/前端基础/JavaScript.md'},
        ],
    },
    {
        text: '前端框架', items: [
            {text: 'Pinia', link: '/前端开发/前端框架/Pinia.md'},
            {text: 'Vite', link: '/前端开发/前端框架/Vite.md'},
            {text: 'Vue2', link: '/前端开发/前端框架/Vue2.md'},
            {text: 'Vue3', link: '/前端开发/前端框架/Vue3.md'},
            {text: 'VueRouter', link: '/前端开发/前端框架/VueRouter.md'},
        ]
    },
    {
        text: '前端进阶', items: [
            {text: 'Babel', link: '/前端开发/前端进阶/Babel.md'},
            {text: 'Node.js', link: '/前端开发/前端进阶/Node.js.md'},
            {text: 'TypeScript', link: '/前端开发/前端进阶/TypeScript.md'},
            {text: 'Webpack', link: '/前端开发/前端进阶/Webpack.md'},
            {text: 'XML', link: '/前端开发/前端进阶/XML.md'},
        ]
    },
];
let backEndNav = [
    {
        text: '中间件', items: [
            {text: 'Netty', link: '/后端开发/中间件/Netty.md'},
            {text: 'Nginx', link: '/后端开发/中间件/Nginx.md'},
            {text: 'RabbitMQ', link: '/后端开发/中间件/RabbitMQ.md'},
        ]
    },
    {
        text: '大数据', items: [
            {text: 'Flink', link: '/后端开发/大数据/Flink 练习.md'},
            {text: 'Hadoop', link: '/后端开发/大数据/Hadoop 练习.md'},
            {text: 'Kafka', link: '/后端开发/大数据/Kafka 练习.md'},
            {text: 'Spark', link: '/后端开发/大数据/Spark 练习.md'},
        ]
    },
    {
        text: '开发知识', items: [
            {text: 'CDN', link: '/后端开发/开发知识/CDN.md'},
            {text: '定时任务与CRON表达式', link: '/后端开发/开发知识/定时任务与CRON表达式.md'},
            {text: '正则表达式', link: '/后端开发/开发知识/正则表达式.md'},
            {text: '设计模式', link: '/后端开发/开发知识/设计模式.md'},
        ]
    },
    {
        text: '操作系统', items: [
            {text: 'Linux', link: '/后端开发/操作系统/Linux.md'},
            {text: '操作系统', link: '/后端开发/操作系统/操作系统.md'},
            {text: '虚拟机', link: '/后端开发/操作系统/虚拟机.md'},
            {text: '计算机网络', link: '/后端开发/操作系统/计算机网络.md'},
        ]
    },
    {
        text: '数据库', items: [
            {text: 'ElasticSearch', link: '/后端开发/数据库/ElasticSearch.md'},
            {text: 'Mongodb', link: '/后端开发/数据库/Mongodb.md'},
            {text: 'MySQL', link: '/后端开发/数据库/MySQL.md'},
            {text: 'Redis', link: '/后端开发/数据库/Redis.md'},
            {text: '数据库原理', link: '/后端开发/数据库/数据库原理.md'},
        ]
    },
    {
        text: '算法', items: [
            {text: 'LeetCode', link: '/后端开发/算法/LeetCode.md'},
            {text: '算法基础', link: '/后端开发/算法/算法基础.md'},
            {text: '算法思想', link: '/后端开发/算法/算法思想.md'},
            {text: '蓝桥杯比赛', link: '/后端开发/算法/蓝桥杯比赛.md'},
        ]
    },
    {
        text: '编程语言', items: [
            {text: 'C++', link: '/后端开发/编程语言/C++.md'},
            {text: 'Java基础', link: '/后端开发/编程语言/Java基础.md'},
            {text: 'Java进阶', link: '/后端开发/编程语言/Java进阶.md'},
            {text: 'MyBatis', link: '/后端开发/编程语言/MyBatis.md'},
            {text: 'Python', link: '/后端开发/编程语言/Python.md'},
            {text: 'Spring', link: '/后端开发/编程语言/Spring.md'},
        ]
    },
    {
        text: '部署工具', items: [
            {text: 'Docker', link: '/后端开发/部署工具/Docker.md'},
            {text: 'Jenkins', link: '/后端开发/部署工具/Jenkins.md'},
            {text: 'Kubernetes', link: '/后端开发/部署工具/Kubernetes.md'},
        ]
    },
];
let academicCourse = [
    {
        text: '学术论文', items: [
            {text: 'LaTeX公式', link: '/学术课程/学术论文/LaTeX公式.md'},
            {text: 'Zotero', link: '/学术课程/学术论文/Zotero.md'},
            {text: '参考文献格式', link: '/学术课程/学术论文/参考文献格式.md'},
            {text: '论文汇报', link: '/学术课程/学术论文/论文汇报.md'},
        ]
    },
    {
        text: '深度学习', items: [
            {text: '想法', link: '/学术课程/深度学习/想法.md'},
            {text: '服务器教程', link: '/学术课程/深度学习/服务器教程.md'},
            {text: '深度学习', link: '/学术课程/深度学习/深度学习.md'},
            {text: '深度学习框架', link: '/学术课程/深度学习/深度学习框架.md'},
        ]
    },
    {
        text: '研究生课程', items: [
            {text: '图论及其应用', link: '/学术课程/研究生课程/图论及其应用.md'},
            {text: '研究生英语', link: '/学术课程/研究生课程/研究生英语.md'},
            {text: '软件体系结构', link: '/学术课程/研究生课程/软件体系结构.md'},
            {text: '软件工程管理', link: '/学术课程/研究生课程/软件工程管理.md'},
            {text: '软件开发方法', link: '/学术课程/研究生课程/软件开发方法.md'},
        ]
    },
    {
        text: '研究生面试', items: [
            {text: '专业笔试', link: '/学术课程/研究生面试/专业笔试.md'},
            {text: '专业面试', link: '/学术课程/研究生面试/专业面试.md'},
            {text: '英语面试', link: '/学术课程/研究生面试/英语面试.md'},
        ]
    },
];
let workEfficiency = [
    {text: 'Git', link: '/工作效率/Git.md'},
    {text: 'PS', link: '/工作效率/PS.md'},
    {text: 'UML', link: '/工作效率/天池比赛.md'},
    {text: '天池比赛', link: '/工作效率/天池比赛.md'},
    {text: '快捷键', link: '/工作效率/快捷键.md'},
];
let projectDoc = [
    {
        text: 'API网关', items: [
            {text: '项目概要', link: '/项目文档/API网关/项目概要.md'},
        ]
    },
    {
        text: '中间件设计', items: [
            {text: 'ORM框架', link: '/项目文档/中间件设计/ORM 框架.md'},
            {text: '中间件概念', link: '/项目文档/中间件设计/中间件概念.md'},
        ]
    },
    {
        text: '土味商城', items: [
            {text: '接口设计规范', link: '/项目文档/土味商城/接口设计规范.md'},
            {text: '接口详细设计', link: '/项目文档/土味商城/接口详细设计.md'},
            {text: '部署笔记', link: '/项目文档/土味商城/部署笔记.md'},
            {text: '需求分析报告', link: '/项目文档/土味商城/需求分析报告.md'},
            {text: '项目总体规划', link: '/项目文档/土味商城/项目总体规划.md'},
            {text: '项目概要', link: '/项目文档/土味商城/项目概要.md'},
        ]
    },
    {
        text: '抽奖系统', items: [
            {text: '待开发', link: '/项目文档/抽奖系统/待开发.md'},
        ]
    },
    {
        text: '沛信', items: [
            {text: '1.概要设计', link: '/项目文档/沛信/1.概要设计.md'},
            {text: '2.通信协议', link: '/项目文档/沛信/2.通信协议.md'},
            {text: '3.数据库设计', link: '/项目文档/沛信/3.数据库设计.md'},
            {text: '4.服务端架构设计', link: '/项目文档/沛信/4.服务端架构设计.md'},
            {text: '5.客户端架构设计', link: '/项目文档/沛信/5.客户端架构设计.md'},
        ]
    },
];

module.exports = [
    {text: '前端开发', items: frontEndNav},
    {text: '后端开发', items: backEndNav},
    {text: '学术课程', items: academicCourse},
    {text: '工作效率', items: workEfficiency},
    {text: '项目文档', items: projectDoc},
];
