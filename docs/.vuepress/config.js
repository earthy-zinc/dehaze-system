module.exports = {
    title: '阅读笔记',
    description: 'Learning Notes for Full Stack Development 全栈开发学习笔记',
    port: 8911,
    themeConfig: {
        nav: [
            { text: '首页', link: '/' },
            {
                text: '前端开发',
                items: [
                    { text: 'Vue3', items: [] },
                    { text: 'CSS', link: '/前端开发/CSS'}
                ]
            },
            {
                text: '后端开发',
                items: [
                    { text: 'Java', items: [] },
                    { text: 'Liunx', link: '/后端开发/Linux'}
                ]
            },
            {
                text: '数据库',
                items: [
                    { text: '数据库原理', link: '/数据库/数据库原理' },
                ]
            },
            {
                text: '机器学习',
                items: [
                    { text: '深度学习', link: '/机器学习/深度学习' },
                ]
            },
            {
                text: '算法',
                items: [
                    { text: '算法', link: '/算法/算法' },
                ]
            }
        ],
        sidebar: 'auto',
        lastUpdated: 'Last Updated',
        smoothScroll: true,
        search: false
    }
}