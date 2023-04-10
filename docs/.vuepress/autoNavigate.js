const fs = require('fs');
const path = require('path');

const document = './docs/';
const rootPath = path.resolve(document);

function getNav(dirPath, arr){
    let addr = fs.readdirSync(dirPath);

    addr.forEach((filename) => {
        // 当前文件绝对路径
        const fileDir = path.join(dirPath, filename);
        // 以同步方式获取当前文件信息
        const stats = fs.statSync(fileDir);

        const isFile = stats.isFile();
        const isDir = stats.isDirectory();

        if (isFile && (path.extname(fileDir) === '.md')) {
            // 处理了多余的绝对路径，第一个 replace 是替换掉rootPath之前的绝对路径
            // 第二个是所有满足\\的直接替换掉
            let nav = {
                text: filename.replace('.md', ''),
                link: fileDir.replace(rootPath, '')
                    .replace(/\\/img, '/')
                    .replace('.md', '')
                    .replace('docs','')
            };
            arr.push(nav);
        }
          // 如果是文件夹
        if (isDir) {
            let child = {
                text: filename,
                items: []
            };
            arr.push(child);
            getNav(fileDir, child.items);
        }
    })
}

module.exports = function filterNav(){
    let nav = [
        { text: '首页', link: '/' }
    ];
    getNav(document, nav);
    return nav.filter((item) => {
        if(item.text === '.vuepress') return false;
        if(item.text === 'README') return false;
        return true;
    });
}
