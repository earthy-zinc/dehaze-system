# Node.js 学习

## 一、Node.js 介绍

## 二、NPM

### 1、介绍

NPM是随Node.js一起安装的包管理工具

* 允许用户从NPM服务器下载别人编写的第三方包到本地使用
* 允许用户从NPM服务器下载并安装别人编写的命令行程序到本地使用
* 允许用户将自己编写的包或者命令行程序上传到NPM服务器供别人使用

### 2、NPM常用命令

#### 1）安装模块

如果想要安装node.js的模块，需要使用以下命令：

```shell
npm install <moduleName> 	#本地安装
npm install <moduleName> -g #全局安装
```

**本地安装**

* 会将安装包放在运行npm命令时所在的目录的 ./node_modules 目录下，如果没有该目录会自动创建一个
* 通过 `require()` 在JavaScript代码中引入该包，如`var moduleName=require("moduleName");`

**全局安装**

* 将安装包放在 user/local （用户文件夹）或者node.js的安装目录
* 安装好后可以直接在命令行使用

#### 2）卸载模块

```shell
npm uninstall <moduleName>
```

#### 3）查看模块

```shell
npm ls		 #查看当前目录下已安装的模块
npm list -g  #查看全局安装的模块
```

#### 4）更新模块

```shell
npm undate <moduleName>
```

#### 5）搜索模块

```shell
npm search <moduleName>
```

#### 6）创建模块

```shell
npm init
# 如果想要发布该模块
npm adduser # 在npm资源库中注册成为新用户
npm publish # 发布你所创建的模块
```

#### 7）其他常用命令

```shell
npm help 			#查看所有的命令
npm help <command>  #查看某条命令的详细帮助
```

### 3、package.json

我们可以把自己写的JS文件做成一个软件包供别人使用，那么 package.json 就是用来识别不同的软件包的一个文件，package.json 位于模块的目录下，也会出现在使用node.js的项目的目录下，用于定义包的属性。常用的属性说明：

* name - 包名
* version - 包的版本号
* description - 包的描述信息
* homepage - 包的官网
* author - 包的作者
* contributors - 包的其他贡献者
* dependencies - 包所依赖的其他包的列表
* devDependencies -  开发时所依赖的软件包列表
* repository - 包代码存放库
* main - 指定程序主入口文件，通常为模块根目录下的 index.js
* private - 说明这是一个私有的软件包，不对外开放，设置为true可以防止软件被意外发布到网上
* keyword - 关键字

### 4、package-lock.json

package-lock.json这个文件用来描述每个依赖的软件包的确切版本，以便最终生成的软件可以被完全的复制，而不受依赖的其他软件包版本更新造成的兼容型问题。

因此我们无需将我们项目所依赖的软件包提交到git上，如果我们运行了npm install初始化项目命令，那么npm会读取package.json和package-lock.json，安装项目依赖的软件包对应的确切版本。

### 5、使用淘宝NPM镜像

   ```shell
   npm install -g cnpm --registry=https://registry.npm.taobao.org
   cnpm install <moduleName> # 使用淘宝镜像安装模块
   ```

### 6、registry

registry意为登记处、注册处，npm为了方便管理，使用登记处（**registry**）来统一管理各方发布的软件包，一般通过名称和版本号来识别不同的软件包，默认情况下我们可以通过npm公共登记处（ **npm public registry** ）上传分享自己的软件包和下载别人的软件包供自己使用，服务器网址在 https://registry.npmjs.org 。我们也可以配置自己的远程库服务器来统一管理软件包。淘宝npm镜像就是一个npm public registry。我们可以通过修改npm的配置来更改软件包从哪个远程库总下载。

### 7、scope

#### 作用域介绍

每个软件包都有名称，有些软件包有它自己的作用域（scope）。作用域通常在软件包名称的前面以@符号为开头，并以 / 符号结束。如 `@scope/packagename`。作用域是分组相关的软件包的一种方式。每一个用户或者公司都有他们自己的软件包作用域，只有他们自己能在作用域内添加软件包。目前npm服务器中存在有作用域的软件包，也存在没有作用域的软件包。这两种类型的软件包可以相互依赖。

#### 下载作用域内的软件包

有作用域的软件包在node_modules中是放在以它的作用域为名称的文件夹里面放着的，如一个作用域名为scope的名为package的软件包在项目中的位置是 ./node_modules/@scope/package。我们可以通过文件夹名称首部的@符号来判断一个软件包在哪个作用域里面，因此如果我们想要下载在某个作用域中的软件包，我们需要以`npm install @scope/package`的形式下载。在package.json文件中，对某个软件包的依赖会以 @scope/package 这种名称的形式存在。

#### 项目文件中引入作用域内的软件包

作用域内的软件包实际上是在放在子文件夹中的，npm默认会引入node_modules文件夹下的软件包，语法是`require('package')`，我们要引入作用域内的软件包的话就需要在包名前面说明它的父文件夹名称，如`require('@scope/package')`。

### 8、config

npm的配置信息是自定义npm运行方式的办法，我们可以通过配置registry来改变npm从哪个远程库中下载软件包等等。

获取配置信息的优先级是以 命令行标志>环境变量>npmrc文件>默认配置 

#### 命令行标志

在命令行中输入`--config description`意思是将名为 config 的变量的值改为description。两道横线`--`告诉命令行解析器停止阅读命令行标志，只输入`--config`而不输入任何值意思是设置值为true。

#### 环境变量

任何以`npm_config_`开头的环境变量都会被看做配置参数。

#### npmrc 配置文件

一般情况下，这些配置文件可能会在四个地方：

## 三、交互式解析器

node.js 交互式解析器 real evaluation print loop(REPL) 是一个类似windows控制台的一个界面，在该界面下输入命令，可以实时的获得响应。适合验证Node.js和JavaScript的相关API。

**REPL命令**

* node - 在控制台输入node进入REPL界面
* CTRL+C / CTRL+D / .exit - 退出当前终端
* 向上 向下键 - 查看输入的历史命令
* TAB - 列出当前命令
* .help - 帮助
* .break / .clear - 退出多行表达式
* .save filename - 保存当前会话到指定文件
* .load filename - 从指定文件载入会话

## 四、模块系统

模块是Node.js的基本组成部分，文件和模块是一一对应的。

Node.js提供了exports require 两个对象，exports是公开模块的接口，require是从外部获取模块的接口，即获取其他模块的exports对象。

**情况1**：向外暴露exports对象，然后通过exports对象调用自己写的对象

```javascript
//module1.js文件
exports.world=function(){
	console.log("nothing");
}
```

```javascript
//module2.js文件
var module_1=require("./module1.js");
module_1.world();
```

模块1向外暴露了名为exports的对象，里面有world方法。也就是说通过exports对象把world这个函数作为该模块的访问接口，那么在模块2中，我们只需要通过require方法加载模块1，就可以载入module1中的exports对象，然后使用exports对象中的方法（即模块1中的world函数）了。

如果我们只想向外暴露我们自己写的对象，而不是名为exports的对象。那就需要使用module.exports，如

**情况2**：向外暴露自己写的对象，直接调用自己写的对象。

```javascript
//module1.js文件
module.exports=function(){
	console.log("nothing");
}
```

```javascript
//module2.js文件
var world=require("./module1.js");
world();
```

## 五、事件触发与监听器

node对于对于异步的回调函数，在执行该函数之后，node不会等待回调函数执行完毕之后才进行下一步操作，它会将这个回调函数移出它的消息队列，而这个回调函数会在一个新线程中执行任务，node就会继续执行代码的下一步操作，如果都已经处理完成的话，他就会查看消息队列中的东西，只有回调函数执行完成任务并返回后，该回调函数才会被放到消息队列中，然后node观察到了这个函数，就会继续执行这个函数。

node所有的异步操作在完成时都会发送一个事件到时间队列，所有产生事件的对象都是EventEmitter实例，EventEmitter是由node.js提供的events模块提供的，它的核心就是事件的触发与事件的监听。我们可以实例化一个EventEmitter对象，然后给它绑定一个事件监听器，同时规定一个回调函数，这个回调函数会在监听到该时间后被触发，然后我们给EventEmitter对象发送该事件，那么监听器就会被触发，并且执行回调函数。我们可以给这个EventEmitter对象绑定多个事件，只要我们向该对象发送这些事件那么对应的监听器就会被触发。我们也可以给一个事件绑定多个监听器。他们会依次被触发。

> Emit v. 动词，意为发射，放射

```js
var EventEmitter = require('events').EventEmitter;
var event = new EventEmitter();
event.on('a_event',function(){
	//事件a的监听器
})
event.emit('a_event')
```

EventEmitter对象提供了多个属性方法，介绍如下：

| 方法                              | 说明                              |
|---------------------------------|---------------------------------|
| addListener(event, listener)    | 为指定的事件添加一个监听器到监听器数组的尾部          |
| on(event, listener)             | 为指定事件注册一个监听器，接收一个字符串的事件名和一个回调函数 |
| once(event, listener)           | 为指定事件注册一个单次的监听器                 |
| removeListener(event, listener) | 移除指定事件的一个监听器，必须是已经注册过的监听器       |
| removeAllLIstener([event])      | 移除所有事件的监听器，也可以只移除单个事件的所有监听器     |
| setMaxListeners(n)              | 设置最大监听器数量                       |
| listeners(event)                | 返回指定事件的监听器数组                    |
| emit(event,[arg1],[arg2]...)    | 按监听器的顺序执行每个监听器                  |

## 六、IO操作

### 1、Buffer

在处理TCP或者文件流的时候，必须要使用到二进制数据，这就需要一个缓冲区，来暂时存放二进制数据流，因此node.js定义了一个buffer类，可以创建一个专门存放二进制数据的缓冲区对象。创建buffer对象建议使用`Buffer.from()`接口。

创建及使用Buffer类对象的方法如下

| 方法                                                              | 说明                                                                           |
|-----------------------------------------------------------------|------------------------------------------------------------------------------|
| `alloc(size[, fill], encoding]])`                               | 返回一个指定大小的缓冲区实例，默认缓冲区内填满0                                                     |
| `allocUnsafe(size)`                                             | 同上，但是不会初始化                                                                   |
| `from(array)`                                                   | 返回一个用数组中的值初始化的一个新的缓冲区实例，缓冲区的大小为数组的大小                                         |
| `from(arrayBuffer[, byteOffset], length]])`                     | 返回一个新的缓冲区实例，但是它与给定的用数组建的缓冲区实例共享同一片内存                                         |
| `from(buffer)`                                                  | 从传入的缓冲区实例中复制一个新的缓冲区实例                                                        |
| `from(string[, encoding])`                                      | 返回一个用字符串值初始化的缓冲区实例，可以指定字符在缓冲区中的编码方式。                                         |
| `write(string[, offset[, length]] [, encoding])`                | 根据字符的编码将字符串的内容写入到缓冲区的指定位置，可以规定写入的字节数，如果缓冲区内没有足够的空间，则只会写入一部分，返回实际写入的大小        |
| `toString([encoding[, start[, end]]])`                          | 从缓冲区中读取数据，可以指定开始读取和结束读取的位置，使用的编码默认为utf-8                                     |
| `toJSON()`                                                      | 将缓冲区的内容转换会JSON对象                                                             |
| `concat(list[, totalLength])`                                   | 合并缓冲区，需要一个数组，该数组的元素是由缓冲区组成，那么该方法将这些缓冲区合并为一个。返回这个新的合并后的缓冲区对象。还可以指定合并后的缓冲区总长度。 |
| `compare(otherBuffer)`                                          | 比较两个缓冲区的相对位置                                                                 |
| `copy(targetBuffer[, targetStart[, sourceStart[, sourceEnd]]])` | 拷贝数据到一个缓冲区的特定位置                                                              |
| `slice([start[, end]])`                                         | 裁剪一个缓冲区                                                                      |
| `length`                                                        | 返回缓冲区的长度                                                                     |

### 2、stream

#### 1）流的概念

流这一词形象的表达了数据的传递方式，在数据结构的课程中我们知道，我们想要的数据被逻辑上分成了好几种结构，顺序表、链表、树、图等结构，而这些结构在通过底层计算机硬件传输的时候，并不能表现出来，也在那种情况下无需表现出来，因此我们暂时抛弃数据中隐含的结构，只把这些数据当做无结构的字节序列或者是字符序列，这些被看作是无结构的数据从一个地方传递到另一个地方，我们把他形象的表示成从一个地方流动到另一个地方，我们就可以说数据是以流的方式进行传输。

#### 2）数据的表现形式

数据在最底层一般是以比特为基本单位的，中层是以字节为基本单位，而在高层是以字符为基本单位的。

>  一字节等于八比特，一个比特就是计算机中的一个信号量。而一个字符根据编码方式的不同，可能占据一个字节或两个字节。比如说汉字占用两个字节/四个字节的大小，阿拉伯数字、英文字母占用一个字节的大小。

在输入输出流中，我们并不是以最底层的比特为基本单位，因为一字节等于八比特已经规定好了，不需要在进行复杂的转换，直接按照预订的逻辑进行操作。输入输出流需要处理的字节和字符之间的转换。输入输出流传递数据的过程，主要分为4个步骤：格式化和解析、缓冲、编码转换、传递

#### 3）输入输出流传递数据的步骤

[IOl流](https://baike.baidu.com/item/IO%E6%B5%81/18864794?fr=aladdin)

#### 4）Node.js Stream

在Node.js中Stream是一个抽象的接口，它并不是一个实际的对象，但是很多对象实现了这个接口，能够让数据以流的形式传输。比如说对HTTP服务器发起请求的request对象、标准的输入输出，就是一个Stream。

所有的实现Stream接口的对象也是EventEmitter实例，因此包含事件触发和监听器。我们把IO操作看做事件，因此可以在IO操作中绑定监听器，在IO操作触发时，实现监听器的功能。常见的事件类型有：

* data - 当有数据可以读的时候
* end - 当没有更多的数据可以读的时候
* error - 接收和写入数据过程中发生错误时
* finish - 当所有的数据已经被写入到底层系统中

## 七、全局对象

JavaScript中有一个特殊对象，叫做全局对象，它的所有属性都可以在程序的任何地方访问，在浏览器中window是全局对象，在node.js中global是全局对象。我们可以直接访问到global的属性。满足以条件的变量就是全局变量：

* 在最外层定义的变量
* 全局对象的属性
* 隐式定义未直接赋值的变量

下面说明一下常用的全局变量：

| 全局变量                                 | 类型     | 说明                                                         |
| ---------------------------------------- | -------- | ------------------------------------------------------------ |
| __filename                               | string   | 当前正在执行的脚本代码的文件名，输出文件文件的绝对路径       |
| __dirname                                | string   | 当前执行脚本代码文件所在的目录                               |
| setTimeout(callbackFunction,millisecond) | function | 在指定的毫秒后执行回调函数，只执行一次，返回该定时器的标识符。 |
| clearTimeout(timer)                      | function | 需要传入一个定时器标识符，停止这个定时器                     |
| setInterval(callback, millisecond)       | function | 以指定的毫秒为间隔循环的执行回调函数，返回该定时器的标识符   |
| clearInterval(timer)                     | function | 需要传入一个定时器标识符，停止这个定时器                     |
| console                                  | object   | 用于提供控制台的标准输出                                     |
| process                                  | object   | 用于描述当前进程状态，提供了与操作系统相关的属性和方法       |

process对象的属性：

| 属性              | 类型 | 说明 |
| ----------------- | ---- | ---- |
| exit([code])      |      |      |
| beforeExit        |      |      |
| uncaughtException |      |      |
| Signal            |      |      |

## 八、常用工具模块

### 1、util

util (utilitation) 是node.js的一个常用工具模块，我们在使用他之前需要先引入它，通过`const util = requrie('util')`引入

| 方法                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| callbackify(original_function)                               | 将ES6中用async修饰的函数（或者是返回Promise的函数）转换为遵循异常优先的回调风格函数。 |
| inherits(constructor, superConstructor)                      | 实现对象间原型继承的函数                                     |
| inspect(object, [bool showHidden],[number depth],[bool colors]) | 将一个任意对象转换为字符串的方法，用于调试和错误输出，接收一个要转换的对象object，showHidden能够输出更多隐藏信息，depth表示对象最大递归的层数，默认为两层，如果对象本身嵌套对象层数超过两层则不显示，Color用于更漂亮的输出效果。 |
| isArray(Object)                                              | 判断对象是否是数组                                           |
| isRegExp(object)                                             | 判断对象是否是正则表达式                                     |
| isDate(object)                                               | 判断对象是否是日期                                           |

### 2、fs

fs 是node.js中的文件系统模块。提供了用于文件操作的相关方法，这些方法有异步和同步的版本。异步方法在IO操作的时候不会阻塞程序的运行。

| 方法                                                | 参数                                                         | 返回值                                   |
| --------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| open(path, flags[, mode], callback)                 | flags指以何种方式打开文件，mode设置文件权限，回调函数有两个参数，error错误信息和fd文件描述符(file description) | 打开一个文件，但是并不是在此处读取文件。 |
| stat(path, callback)                                | 回调函数有两个参数，error错误信息和stats是一个对象           | 获取文件信息                             |
| writeFile(file, data[, options], callback)          | file文件名或者文件描述符，data为要写入的数据，可以是字符串或者缓冲区对象，可选参数option是一个对象，包含encoding mode flag三个属性，回调函数只有错误信息参数 | 将数据写入文件                           |
| read(fd, buffer, offset, lenth, position, callback) | 与open方法配合使用，fd为文件描述符，buffer缓冲区对象，offset缓冲区写入数据的偏移量，length要从文件中读取的字节数，length文件读取的起始位置，回调函数有三个参数，err，bytesRead读取的字节数，buffer缓冲区对象。 | 读取文件信息到buffer中                   |
| close(fd, callback)                                 | 与open方法配合使用                                           | 关闭文件                                 |
| ftruncate(fd, len, callback)                        | len文件内容的截取长度，超过的部分会被逻辑上删除              | 截取文件                                 |
| unlink(path, callback)                              |                                                              | 删除文件                                 |
| mkdir(path[, option], callback)                     | option参数recursive递归的创建目录，mode设置目录权限          | 创建目录                                 |
| readdir(path, callback)                             |                                                              | 读取目录信息                             |
| rmdir(path, callback)                               | 回调函数中没有参数                                           | 删除目录                                 |

stats对象方法，通过获取文件信息回调中的stats获取stats对象：

| 方法             | 说明 |
| ---------------- | ---- |
| isFile()         |      |
| isDirectory      |      |
| isBlockDevice()  |      |
| isSymbolicLink() |      |
| isFIFO           |      |
| isSocket()       |      |

### 3、url



### 4、http

http模块是node.js的网络核心模块。

| 属性         | 说明 |
| ------------ | ---- |
| METHODS      |      |
| STATUS_CODES |      |
| globalAgent  |      |

| 方法           | 说明 |
| -------------- | ---- |
| createServer() |      |
| request()      |      |
| get            |      |
|                |      |
|                |      |
|                |      |

同时http模块还提供了五个类。下面分别介绍这五个类。他们都是对象，

| 类                   | 说明 |
| -------------------- | ---- |
| http.Agent           |      |
| http.ClientRequest   |      |
| http.Server          |      |
| http.ServerResponse  |      |
| http.IncomingMessage |      |



### 5、os

提供了操作系统的相关信息。`var os =require('os')`

| 方法                | 描述                     |
| ------------------- | ------------------------ |
| tmpdir()            | 操作系统默认的临时文件夹 |
| endianness()        | CPU的字节序              |
| hostname()          | 主机名                   |
| type()              | 操作系统名               |
| platform()          | 编译时操作系统名称       |
| arch()              | CPU架构                  |
| release()           | 操作系统发行版本         |
| uptime()            | 操作系统运行时间         |
| loadavg()           | 操作系统平均负载         |
| totalmem()          | 系统内存总量             |
| freemem()           | 系统空闲可用内存         |
| cpus()              | 每个CPU的信息            |
| networkInterfaces() | 网络接口列表             |

### 6、path

可以用于处理文件路径。` var path =require('path')`

| 方法                     | 说明 |
| ------------------------ | ---- |
| normalize(path)          |      |
| join(path1, path2, ...)  |      |
| resolve([from ... ,] to) |      |
| isAbsolute(path)         |      |
| relative(from, to)       |      |
| dirname(path)            |      |
| basenaem(p[, extetion])  |      |
| extname(path)            |      |
| parse(pathString)        |      |
| format(pathObject)       |      |

### 7、net

用于底层的网络通信，包含了创建服务器和客户端的方法。通过`var net = require('net')`引入，偏底层，平常使用http模块创建服务器更方便。

| 方法                                                  | 描述 |
| ----------------------------------------------------- | ---- |
| `createServer([options] [, connectionListener])`        |      |
| `createConnection(options[, connectionListener])`       |      |
| `createConnection(port[, host] [, connectionListener])` |      |
| `createConnection(path[, connectionListener])`          |      |
| `connect(options[, connectionListener]) `               |      |
| `connect(port[, host] [, connectionListener])`          |      |
| `connect(path[, connectionListener])`                   |      |
| `isIP(input) / isIPv4(input) / isIPv6(input)`           |      |

创建完服务器之后，会生成一个Server对象，有如下几个方法

| 方法 | 说明 |
| ---- | ---- |
|      |      |

同时会产生以下事件：

| 事件 | 说明 |
| ---- | ---- |
|      |      |

创建TCP连接之后，会生成一个socket对象，有如下几个事件：

| 事件 | 说明 |
| ---- | ---- |
|      |      |

属性和方法：

| 属性或方法 | 说明 |
| ---------- | ---- |
|            |      |

### 8、dns

DNS用于解析域名，使用`var dns= require('dns')` 引入

| 方法 | 描述 |
| ---- | ---- |
|      |      |

### 9、domain



### 10、express











