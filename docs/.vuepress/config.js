const getNav = require('./autoNavigate');

module.exports = {
    title: '土味锌阅读笔记',
    description: 'Learning Notes for Full Stack Development 全栈开发学习笔记',
    port: 8911,
    themeConfig: {
        nav: getNav(),
        sidebar: 'auto',
        lastUpdated: '文章最后更新日期',
        smoothScroll: true,
        search: false
    }
}