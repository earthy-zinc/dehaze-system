# JavaScript

## JavaScript语法基础

### 常量与变量

JavaScript中值分为字面量和变量两种：

- **字面量**：固定值，如数字、字符串等
- **变量**：可变化的值

JavaScript数据类型包括：
- 数字：整数、小数、科学记数法
- 字符串：单引号或双引号包裹的文本
- 布尔值：true或false
- 数组：中括号[]包裹的数据集合
- 对象：大括号{}包裹的键值对集合
- 函数：可执行的代码块

### 变量声明关键字

| 关键字 | 特点 | 作用域 |
|-------|------|--------|
| var | 可重复声明，有变量提升 | 函数作用域 |
| let | 不可重复声明，无变量提升 | 块级作用域 |
| const | 声明常量，必须初始化 | 块级作用域 |

### 作用域

1. **局部作用域**：函数内部声明的变量，只能在函数内访问
2. **全局作用域**：函数外部定义的变量，全局可访问
3. **块级作用域**：由{}包裹的代码块形成的作用域

变量生命周期：
- 局部变量：函数执行时创建，函数结束时销毁
- 全局变量：页面加载时创建，页面关闭时销毁

### 函数

#### 函数定义方式

1. **函数声明**
   ```javascript
   function name(parameters) {
       // 函数体
   }
   ```

2. **函数表达式**
   ```javascript
   var x = function(a, b) {
       return a * b;
   };
   ```

3. **箭头函数**
   ```javascript
   // 完整写法
   (参数) => {函数体}
   
   // 简写形式
   单一参数 => 单一表达式
   ```

4. **构造函数**
   ```javascript
   var myFunction = new Function("a", "b", "return a*b");
   ```

#### 函数调用方式

- 事件触发调用
- 直接调用：`funcName()`
- 自调用：`(function (){})();`

#### 函数特性

- 函数提升：函数声明会被提升到作用域顶部
- 函数是对象，具有属性和方法
- 函数可作为构造器创建对象实例

### 对象

#### 对象基本概念

对象是带有属性和方法的特殊数据类型。JavaScript中几乎所有对象都是Object类型的实例。

访问方式：
- 属性访问：`object.attribute`
- 方法调用：`object.functionName()`

#### 对象创建方式

1. **字面量方式**
   ```javascript
   var obj = {name: "value", name2: "value2"};
   ```

2. **构造函数方式**
   ```javascript
   function Person(name) {
       this.name = name;
   }
   var p = new Person("Amy");
   ```

#### ES6对象简写

```javascript
// 属性简写
const age = 12, name = "Amy";
const person = {age, name};

// 方法简写
const person = {
    say() {
        console.log("hi");
    }
};
```

### 数据类型

JavaScript数据类型分为基本类型和引用类型：

**基本类型**：
- string（字符串）
- number（数字）
- boolean（布尔值）
- null（空值）
- undefined（未定义）
- symbol（符号）

**引用类型**：
- Object（对象）
- Array（数组）
- Function（函数）
- RegExp（正则表达式）
- Date（日期）

使用typeof操作符可查看变量数据类型。

### this关键字

this表示当前对象的引用，其指向根据上下文环境而定：

1. 方法中：指向方法所属的对象
2. 全局环境：指向全局对象（严格模式下为undefined）
3. 事件中：指向接收事件的元素
4. call()/apply()：可指定this指向

### 解构赋值

解构赋值是ES6特性，用于从数组或对象中提取值并赋给变量。

#### 数组解构
``javascript
let [a, b, c] = [1, 2, 3]; // a=1, b=2, c=3
```

#### 对象解构
``javascript
let {name, age} = {name: "Amy", age: 12};
```

### 严格模式

在代码或函数顶部添加`"use strict";`启用严格模式，增强错误检查，禁止使用未声明变量。

### 模块化

ES6模块系统分为导出(export)和导入(import)，自动开启严格模式。

#### 导出模块
```javascript
// 分别导出
export let name = "tom";

// 统一导出
let name = "tom";
let say = function() { console.log("hi"); };
export {name, say};

// 默认导出
export default {
    name: "tom",
    say: function() { console.log("hi"); }
};
```

#### 导入模块
```javascript
// 导入指定内容
import {name} from "./module.js";

// 导入默认内容
import m from "./module.js";

// 全部导入
import * as newName from "./module.js";
```

## 浏览器对象模型(BOM)

### Window对象

Window对象表示浏览器窗口，包含以下常用属性和方法：

| 属性 | 说明 |
|-----|-----|
| closed | 窗口是否已关闭 |
| document | 对Document对象的引用 |
| innerHeight/innerWidth | 窗口文档显示区域尺寸 |
| outerHeight/outerWidth | 窗口整体尺寸 |

| 方法 | 说明 |
|-----|-----|
| alert() | 显示警告框 |
| confirm() | 显示确认框 |
| prompt() | 显示输入框 |
| setTimeout()/clearTimeout() | 延时执行/清除 |
| setInterval()/clearInterval() | 定时执行/清除 |
| open()/close() | 打开/关闭窗口 |

### Navigator对象

包含浏览器相关信息。

### Screen对象

包含显示器屏幕信息。

### History对象

管理浏览器历史记录。

### Document对象

Document对象是HTML文档的根节点，用于访问和操作页面元素。

## 文档对象模型(DOM)

### DOM节点类型

1. 文档节点：整个HTML文档
2. 元素节点：HTML元素
3. 属性节点：HTML属性
4. 文本节点：元素中的文本
5. 注释节点：HTML注释

### DOM元素操作

#### 查找元素

- getElementById()：通过ID查找
- getElementsByTagName()：通过标签名查找
- getElementsByClassName()：通过类名查找

#### 创建元素

- createElement()：创建元素节点
- createTextNode()：创建文本节点
- createAttribute()：创建属性节点

### DOM事件

#### 鼠标事件

| 事件 | 触发条件 |
|-----|---------|
| onclick | 鼠标点击 |
| ondblclick | 鼠标双击 |
| onmousedown | 鼠标按键按下 |
| onmouseup | 鼠标按键释放 |
| onmouseenter | 鼠标进入元素 |
| onmouseleave | 鼠标离开元素 |

#### 键盘事件

| 事件 | 触发条件 |
|-----|---------|
| onkeydown | 键盘按键按下 |
| onkeyup | 键盘按键释放 |
| onkeypress | 键盘按键按下并释放 |

#### 表单事件

| 事件 | 触发条件 |
|-----|---------|
| onfocus | 元素获得焦点 |
| onblur | 元素失去焦点 |
| onchange | 表单元素内容改变 |
| onsubmit | 表单提交 |

### 控制台对象

浏览器控制台用于调试，常用方法包括：

| 方法 | 说明 |
|-----|-----|
| console.log() | 输出信息 |
| console.warn() | 输出警告信息 |
| console.error() | 输出错误信息 |
| console.group() | 创建信息分组 |
| console.time()/timeEnd() | 计时 |
| console.clear() | 清除控制台 |

## 异步编程

### 同步与异步

- **同步**：代码按顺序执行，后续代码需等待当前代码执行完毕
- **异步**：不阻塞后续代码执行，通过回调处理结果

### Ajax

Ajax（Asynchronous JavaScript and XML）用于在不刷新页面的情况下与服务器交换数据。

使用XMLHttpRequest对象实现Ajax：

```javascript
// 1. 创建对象
var httpRequest = new XMLHttpRequest();

// 2. 打开请求
httpRequest.open(method, url, async);

// 3. 发送请求
httpRequest.send(data);

// 4. 处理响应
httpRequest.onreadystatechange = function() {
    if (httpRequest.readyState === 4 && httpRequest.status === 200) {
        // 处理响应数据
        var data = httpRequest.responseText;
    }
};
```

### Promise

Promise是处理异步操作的对象，具有三种状态：

1. Pending（进行中）
2. Fulfilled（已成功）
3. Rejected（已失败）

```javascript
var promise = new Promise(function(resolve, reject) {
    if (/* 操作成功 */) {
        resolve('成功数据');
    } else {
        reject('错误信息');
    }
});

promise
    .then(function(value) {
        // 处理成功结果
        return value;
    })
    .catch(function(error) {
        // 处理错误
    })
    .finally(function() {
        // 最终处理
    });
```

### Async/Await

Async/Await是基于Promise的语法糖，使异步代码看起来像同步代码：

```javascript
async function asyncFunction() {
    try {
        let result = await promiseObject;
        return result;
    } catch (error) {
        // 处理错误
    }
}
```

### Axios

Axios是基于Promise的HTTP客户端，用于发送异步请求：

```javascript
// 基本用法
axios.get('/api/data')
    .then(response => {
        // 处理响应
    })
    .catch(error => {
        // 处理错误
    });

// 配置请求
axios({
    method: 'post',
    url: '/api/data',
    data: {key: 'value'}
});
```

