const getNav = require('./autoNavigate');

module.exports = {
    title: '土味锌的阅读笔记',
    description: 'Learning Notes for Full Stack Development 全栈开发学习笔记',
    port: 8911,
    locales: {
        // 键名是该语言所属的子路径
        // 作为特例，默认语言可以使用 '/', 作为其路径。
        '/': {
          lang: 'zh-CN', // 将会被设置为 <html> 的 lang 属性
        }
    },
    themeConfig: {
        nav: getNav(),
        sidebar: 'auto',
        lastUpdated: '文章最后更新日期',
        smoothScroll: true,
        search: false
    }
}