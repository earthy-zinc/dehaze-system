const fs = require('fs');
const path = require('path');

//获取最初给定文件夹的绝对路径
const rootPath = path.resolve('../');

const fileDisplay = async function fileDisplay(dirPath, arr) {
  // 获取给定文件夹的绝对路径
  const filePath = path.resolve(dirPath);
  
  //根据文件路径读取文件，返回文件列表
  fs.readdir(filePath, (err, files) => {
    
    files.forEach((filename) => {
      //获取当前文件的绝对路径
      const fileDir = path.join(filePath, filename);

      // fs.stat(path)执行后，会将stats类的实例返回给其回调函数。
      fs.stat(fileDir, (eror, stats) => {
        const isFile = stats.isFile();
        const isDir = stats.isDirectory();

        if (isFile && (path.extname(fileDir) === '.md')) {
          // 处理了多余的绝对路径，第一个 replace 是替换掉那个路径，第二个是所有满足\\的直接替换掉
          let nav = {text: filename, link: fileDir.replace(rootPath, '').replace(/\\/img, '/')}
          arr.push(nav);
        }
        // 如果是文件夹
        if (isDir) {
          let child = {text: filename, item: []};
          arr.push(child);
          fileDisplay(fileDir, child.item);
        }
      })

    });
  });
}


const arr = [];
fileDisplay('../', arr);
console.log(arr, "arr")