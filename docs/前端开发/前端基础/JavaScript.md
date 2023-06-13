# JavaScript 练习

## 一、JavaScript语法

### 1、常量与变量

在JavaScript中，固定值称为字面量。可以变化的值称为变量。字面量和变量可以是以下这些类型。

* 数字：可以是小数、整数、科学记数法
* 字符串：用单引号或双引号包裹起来
* 表达式：用于计算
* 数组：中括号包裹起来
* 对象：大括号包裹起来，有属性和值
* 函数：通过function 关键字定义一个函数

### 2、关键字

* var 关键字：可以使用var声明变量，变量声明之后是空的undefined，需要赋值，也可以在声明时赋值。在ES6之前是没有块级作用域的概念的，因此var 关键字声明的变量不会被块级作用域所束缚，仍然能在块级作用域外被访问到。尽量使用let声明变量

* let 关键字：let声明的变量只在let命令所在的代码块内有效。

* const 关键字：const声明一个只读的常量，const常量必须在声明时赋值，一旦声明，常量的值不得改变。

### 3、作用域

* 局部作用域：变量在函数内部声明，只能在函数内部访问，该变量为局部作用域。局部变量在函数开始执行时创建，在函数执行完毕后自动销毁。特别的，函数的参数是局部变量。
* 全局作用域：变量在函数外部定义，为全局变量，全局变量具有全局作用域。网页中所有的JavaScript和函数都可使用。如果我们为一个没有声明的变量赋值，那么这个变量会自动成为全局变量。
* 变量生命周期：变量在声明是初始化，局部变量在函数执行结束后销毁，全局变量在页面关闭时销毁。
* 在HTML中，全局变量是window对象，所有的数据变量都属于window对象。
* 声明提升：JavaScript中的函数及用var定义的变量的声明会被提升到最顶部。但是初始化不会，意味着如果函数声明并初始化合并在一起时，因为只有声明被升到头部，初始化并没有。在这条声明并初始化的语句之前访问该变量的结果是undefined。

### 3、函数

#### 1）普通函数

**函数的定义：**函数是由事件驱动或者当它被调用时执行的可重复代码块。

**函数的参数：**调用函数时，可以向函数中传递值，这些值称为参数。函数的形参parameters，指的是在函数定义时所列出的参数名称。函数的实参argument，指的是传递到函数或者是函数接受到的真实的值。

* 参数可以在函数中使用
* 可以向函数中传递任意个数的参数，用逗号分割
* 函数中的参数为局部变量，只能在函数体内部使用
* 函数接受到的所有参数可以通过函数内建的argument对象来读取。argument对象是一个类数组的对象，我们可以通过索引获取对应参数的值，可以通过length属性获取传入的参数总和。

**普通函数的声明：**

```js
function name(parameters){
    //函数代码块
}
```

#### 2）函数表达式

**函数表达式的概念：**JavaScript函数可以通过一个表达式来定义，函数表达式可以存储在一个变量中，在函数表达式存储在变量中后，变量也可以作为一个函数使用。

**函数表达式的写法：**

```js
var x = function (a, b){
    return a*b
};
```

#### 3）匿名函数

**匿名函数的概念：**在函数表达式存储在变量中后，变量也可以作为一个函数使用。此时可以将函数定义为匿名函数，不需要函数名称，通过变量名来调用。

**匿名函数的声明：**`function (){	}`

#### 4）箭头函数

* 写法1：`(参数) => {函数声明}`
* 写法2：`单一参数 => 单一表达式`
* 没有参数的函数：`() => {函数声明}`
* 使用箭头函数时，箭头函数会默认绑定外层this的值。箭头函数没有变量提升，无法先使用后定义。

#### 5）函数构造器

函数也可以通过一个名为Function的函数来创建。该函数接收多个参数，分别是要创建的函数的形参，函数体代码。返回一个函数。

```js
/*创建了一个匿名函数，并将该函数赋给了变量myFunction。
  等价于：var myFunction = function(a, b){ return a*b };
*/
var myFunction = new Function("a","b","return a*b");

//调用函数myFunction并用变量x接受函数的返回值
var x= myFunction(4,3);
```

#### 6）对象构造器

我们可以通过函数创建一个对象。对象构造器和普通的函数没有太多的不同，与普通函数的区别是调用的方法不一样，通过函数构造器来生成一个对象需要使用new关键字。在通常情况下，在函数体内部的this指的是函数的拥有者，即外部的this，是window对象。在严格模式下，this未定义。但是通过函数构造器方法生成对象时，this指的是实例对象本身。如：

```js
function Person(firstName, LastName, callback){
    this.firstName = firstName;
    this.LastName  = lastName;
    this.say = callback;
}
var p = new Person("xiao","wang",function (){
    return this.firstName;
});
```

上述代码通过对象构造器创建了一个Person类型的实例对象。该实例对象有两个属性分别是firstName, lastName。同时该对象拥有一个对象方法say，该方法是一个回调函数，回调函数是在创建该对象是传入的，并没有在构造函数中定义，在创建实例对象之后，我们就可以调用该回调函数。返回设定好的结果，这样做的方式能让我们对对象方法的定义延迟到创建对象的时候。由调用者自己决定执行函数的 具体逻辑。

#### 7）函数的调用

* 事件发生时：为某个JavaScript事件定义一个函数，当用户点击按钮时，或者执行某种操作时，函数会被调用。
* 程序员通过JavaScript代码调用：在定义函数之后，我们使用该函数名称+()来手动调用该函数，如`funcName();`
* 自调用：函数表达式可以自调用，`(function (){})();`对函数声明本体加一个括号，把他作为一个函数表达式，后面再加一个括号可以实现函数自调用，假如该函数只需执行一次，可以考虑使用这种方法。

#### 8）函数的性质

* 函数提升：变量的声明和函数的声明都将提升到当前作用域之前，因此函数可以在声明前调用。
* 函数是一个对象，有自己的属性和方法。比如说argument属性可以接收到传给函数的参数，toString方法将函数定义转换为字符串。
* 函数如果用于创建新的对象，称为对象的构造函数。
* 函数的返回值也可以赋给一个变量。注意要和函数表达式的写法做出区分。在函数表达式中，变量是函数本身。把函数返回值赋给一个变量时，该变量是函数返回值的类型。

#### 9）函数的方法

##### call()

##### apply()

#### 10）闭包



### 4、对象

#### 1）对象的基本概念

对象是带有属性和方法的特殊数据类型。布尔型、数字型、字符串、日期、数学和正则表达式、数组、函数都可以是一个对象。

我们通过`object.attribute`来访问对象的属性，通过`object.functionName()`来调用对象方法。

定义在全局中的函数，this指向该函数的拥有者。而在对象方法中的this指的是该对象实例。

#### 2）创建对象

创建新对象有两种不同的方法：

* 使用object定义并创建对象的实例
* 使用函数定义对象，然后创建新的对象实例

##### 使用Object 来创建对象

在JavaScript中，几乎所有对象都是Object类型的实例，它们都从Object.prototype 中继承属性和方法。

Object 构造函数创造一个对象包装器，会根据给定的参数创建对象：

* 如果给定的值是Null或者undefined，将会创建并返回一个空对象
* 如果传进去的是一个基本类型的值，则会构造其包装类型的对象
* 如果传进去的是引用类型的值，仍然会返回这个值，然后呢？

语法：

1. 通过Object创建对象 `var a = new Object()`
2. 使用对象字面量创建对象 `var a = {name : value, name2 : value2}`

在创建对象之后我们可以向新对象中添加属性：

* `a.attribute="attribute"`

##### 对象属性和方法的简写

在ES6中允许对象的属性简洁的表示。方法名也可以简写。如下所示

```javascript
/************属性的简写*******************/
const age=12,name="amy";
//1.通常方法
const person={age:age, name:name};
//2.简写方法
const person={age, name};

/************方法的简写*******************/
//1.通常方法
const person={
    say:function(){
		console.log("hi");
    }
}
//2.简写方法
const person={
    say(){
		console.log("hi");
    }
}
```

ES6也允许表达式作为属性名，需要将表达式放到方括号中

```javascript
const a="my",b="name";
const person={
    [a+b]:"amy"
}
```

##### 使用对象构造器（构造函数）创建对象

下面通过函数来创建了一个person的对象构造器，其中有属性和方法。我们可以通过`var myFather=new person("john");`创建一个新的对象实例。

```javascript
function person(name){
    this.name=name;
    function changeName(newName){
        this.name=newName;
    }
}
```

##### 其他注意事项

* 我们可以使用for in 循环来遍历对象的属性。
* 对象是可变的，我们可以通过引用传递对象。如`var x = person;` x 和 person 指的是同一个对象。
* 已存在构造器的对象是不能添加新属性的。

#### 3）原型对象

所有JavaScript对象都会从一个原型对象prototype中继承属性和方法。例如Date对象从Date.prototype 继承，Person 对象从 Person.prototype 继承。

而所有JavaScript对象都是位于原型链顶端的Object的实例，JavaScript对象中有一个指向原型对象的链，当试图访问对象的一个属性时，它会先从对象本身搜寻，然后搜寻该对象的原型，以及原型的原型，直到找到一个名字匹配的属性或者找到原型链的末尾。

因此如果想要给已存在构造函数的对象添加新的属性和方法，我们就需要向它的原型添加。使用对象的prototype属性添加新的属性和方法。而对于实例对象，我们需要通过隐式原型`__proto__`找到其原型对象。

### 5、数据类型

JavaScript具有两类数据类型，分别是基本类型和引用数据类型分：

* 基本类型：字符串string、数字number、布尔boolean、空Null、未定义undefined、symbol
* 引用数据类型：对象、数组、函数、正则、日期

JavaScript具有动态数据类型，意味着相同的变量可以用作不同的类型。

变量的数据类型可以使用 typeof 操作符查看

#### 1）基本类型

* **symbol** 是一种基本的数据类型，每一个symbol都是独一无二的。可以接受一个字符串作为symbol的描述，但字符串本身不作为区分不同symbol的标志。
* 

#### 2）引用类型







### 6、this

this 表示的是当前对象的一个引用。JavaScript中的this不是固定不变的。

* 在方法中，this表示该方法所属的对象
* 单独使用，this表示全局对象
* 函数中，this表示全局对象。而在严格模式下，this是未定义的undefined
* 在事件中，this表示接收该事件的元素。
* call()和apply()方法可以将this引用到任意对象

### 7、解构赋值

解构赋值就是分解并赋值数组或者对象的结构到不同的变量中去。解构了赋值表达式左边的部分，可以根据右值将左边的部分分为好几个变量。

官方说法是解构赋值是针对数组和对象进行模式匹配，然后对其中的变量进行赋值。

#### 1）数组模型的解构

* 基本解构：`let [a,b,c]=[1,2,3]`，那么接下来a=1, b=2, c=3
* 可嵌套解构：
* 可忽略：
* 不完全解构：
* 不均等解构：
* 解构字符串：

#### 2）对象模型的解构



### 8、严格模式

在代码文件的头部或者是在函数的头部使用`"use strict";`声明。在严格模式下，不能使用未声明的变量

### 9、模块

ES6的模块化分为导出和导入两个模块（export import）自动开启严格模式，在模块中可以导入导出各种类型的变量，如函数、对象、字符串、数字、布尔值、类等。每个模块都有自己的上下文，在模块内部声明的变量都是局部变量。每个模块只加载一次。

#### 1）导出模块

export 命令有如下的特点：

* 在通常情况下导出的函数声明和类声明必须要有**名称**
* 不仅能够导出声明还能够导出引用
* export 命令可以出现在模块的任何位置，但必须位于模块顶层

export default 命令有如下的特点：

* 在一个文件中，export、import命令可以有多个，但是export default 命令只能有一个
* export default 中的default对应的是导出接口的变量
* 通过export 导出。导入是需要加上 { }，而export default 不需要
* export 向外暴露的成员，可以使用任意变量接收

```javascript
/**********导出****************/
//用法1：分别暴露
export let name="tom";

//用法2：统一暴露
let name="tom";
let say = function(){
    console.log("hi");
}
export { name, say }

//用法2:默认暴露，此时暴露的是一个对象，不再需要自己命名，对象名称default代替
export default {
    name:"tom",
    say:function(){
    	console.log("hi");
    }
}
```

export 命令导出的变量的所用名称，需要和模块内部变量名称一一对应。

import 命令导入变量所用的名称，需要和导出接口所用的名称相同，但变量之间的顺序可以改变。

为了预防重名的现象发生，在统一导入导出接口中，我们可以使用 as 重新定义变量的名称。

#### 2）导入模块

import命令的特点：

*　只读的：不能改变import 中变量类型的值
*　单例：多次重复执行import语句，只会执行一次
*　静态的：不能使用表达式，以及预先声明好的变量
*　import 命令会提升到整个模块顶部，首先执行

在HTML中使用模块化JavaScript，需要使用`<script type="module">`。

```javascript
/**********导入****************/
//对于分别暴露和统一暴露，{}里面的名称需要和export中一致
import {name} from "./module1.js"

//对于默认暴露，名称可以任意，而且不需要加{}
import m from "./module3.js"

//全部导入，需要进行命名。对于三种暴露都适用，返回的是一个对象
import * as newName from "./module3.js"
```

## 三、窗口对象

### 1、window对象

包含了当前窗口相关的信息

| 属性        | 说明                                   |
| ----------- | -------------------------------------- |
| closed      | 返回窗口是否关闭                       |
| frames      | 返回窗口中所有命名的框架               |
| innerHeight | 返回窗口文档显示区的高度               |
| innerWidth  | 返回窗口文档显示区的宽度               |
| outerHeight | 返回窗口的外部高度，包括滚动条和状态栏 |
| outeWidth   | 返回窗口的外部宽度，包括滚动条和状态栏 |
| document    | 对document对象的只读引用               |
| navigator   | 对navigtor对象的只读引用               |
| screen      | 对screen对象的只读引用                 |
| history     | 对history对象的只读引用                |
| location    | 对location对象的只读引用               |

| 方法                           | 说明                                                    |
| ------------------------------ | ------------------------------------------------------- |
| alert()                        | 显示带有一段消息和一个确认按钮的警告框                  |
| confirm()                      | 显示带有一段消息和确认按钮和取消按钮的对话框            |
| prompt()                       | 显示可提示用户输入的对话框                              |
| getSelection()                 | 返回一个selection对象，表示用户选择的文本范围或光标位置 |
| getComputedStyle()             | 获取指定元素的CSS样式                                   |
| moveBy()                       | 按指定的距离来移动窗口                                  |
| moveTo()                       | 把窗口移动到指定的位置                                  |
| resizeBy()                     | 按照指定像素调整窗口大小                                |
| resizeTo()                     | 把窗口大小调整为指定的宽度和高度                        |
| setInterval(), clearInterval() | 按照指定的间隔循环调用函数或计算表达式，停止调用        |
| setTimeout(),cleatTimeout()    | 在指定的毫秒数调用函数或计算表达式，停止调用            |
| open()                         | 打开一个新的浏览器窗口                                  |
| close()                        | 关闭当前浏览器窗口                                      |
| print()                        | 打印当前窗口                                            |

### 2、Navigator对象

包含了当前浏览器有关的信息

### 3、Screen对象

包含了显示器屏幕有关的信息

### 4、history对象

包含了用户在当前窗口访问过的URL，访问的历史记录

### 5、document对象

内容较多，详见下一章

## 四、文档对象

### 1、HTML DOM Document 对象

在HTML DOM 中，每一个元素都是节点，文档是一个文档节点，HTML元素是一个元素节点，HTML属性是一个属性节点，HTML元素中间的文本是文本节点，注释是注释节点。当浏览器载入HTML文档时，文档会成为document对象，document对象是HTML文档的根节点，我们可以从根节点出发对HTML页面中的所有元素进行访问。而document对象是window对象的一部分。

| 属性/方法                   | 描述                                |
| --------------------------- | ----------------------------------- |
| document.activeElement      | 返回当前获取的焦点元素              |
| document.addEventListener() | 向文档中添加事件                    |
| document.anchors            | 返回文档中所有超链接元素`<a>`的引用 |
| document.baseURI            |                                     |
| document.body               |                                     |
| document.cookie             |                                     |
| createAttribute()           |                                     |
| createComment()             |                                     |
| creatElement()              |                                     |
| createTextNode()            |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |
|                             |                                     |



### 2、HTML DOM 元素对象

| 属性/方法 | 描述 |
| --------- | ---- |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |
|           |      |



### 3、HTML DOM 属性对象



### 4、HTML DOM 事件对象

#### 1）鼠标事件

| 属性          | 描述                                    | 实例                                           |
| ------------- | --------------------------------------- | ---------------------------------------------- |
| onclick       | 当用户点击某个对象时                    | `<button onclick="myFunction()">点我</button>` |
| oncontextmenu | 当用户点击鼠标右键打开上下文菜单时      | `<div oncontextmenu="myfunction()">点我</div>` |
| ondblclick    | 当用户鼠标双击某对象时(on double click) |                                                |
| onmousedown   | 鼠标按键被按下                          |                                                |
| onmouseup     | 鼠标按键被松开                          |                                                |
| onmouseenter  | 鼠标移动到某个元素上                    |                                                |
| onmouseleave  | 鼠标移出某个元素                        |                                                |
| onmousemove   | 鼠标被移动                              |                                                |
| onmouseover   | 鼠标移动到某个元素上                    |                                                |
| onmouseout    | 鼠标从某元素移开                        |                                                |

#### 2）键盘事件

| 属性       | 描述               | 实例 |
| ---------- | ------------------ | ---- |
| onkeydown  | 当键盘被按下       |      |
| onkeyup    | 当键盘被松开       |      |
| onkeypress | 当键盘被按下并松开 |      |

#### 3）表单事件

| 属性       | 描述                   | 实例 |
| ---------- | ---------------------- | ---- |
| onfocus    | 获取焦点时             |      |
| onblur     | 失去焦点时             |      |
| onfocusin  | 即将获取焦点时         |      |
| onfocusout | 即将失去焦点时         |      |
| onchange   | 表单元素内容发生改变时 |      |
| oninput    | 输入时                 |      |
| onreset    | 重置表单时             |      |
| onsearch   | 搜索时                 |      |
| onselect   | 选择文本时             |      |
| onsubmit   | 提交表单时             |      |

#### 4）剪切板事件

| 属性 | 描述 | 实例 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

#### 5）打印事件

| 属性 | 描述 | 实例 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |

#### 6）拖动事件



#### 7）多媒体事件



#### 8）其他事件



### 5、HTML DOM 控制台对象console

| 方法                                  | 效果                                       |
| ------------------------------------- | ------------------------------------------ |
| log(), info(),warn(),error()          | 控制台输出一条信息                         |
| group(), groupCollapsed(), groupEnd() | 在控制台创建一个信息分组，以groupEnd结束。 |
| time(), timeEnd()                     | 计时，以timeEnd结束，算出操作花费的时间    |
| trace()                               | 显示当前代码在堆栈中调用路径               |
| clear()                               | 清除控制台的信息                           |
| count()                               | 记录调用次数                               |
| assert()                              | 断言                                       |



## 五、异步编程

### 1、同步与异步

单线程编程中，程序的运行是同步的，代码从前往后按照顺序执行，这是同步的概念。而异步是新建一个线程去处理该请求，那么主线程在处理结果出来之前就可以去做其他的事情，这种情况下不保证代码按照先后顺序执行。

我们常用异步编程去处理一些耗时较长、可以后台运行的任务，比如说下载资料、发出网络请求，用户在发出这些请求之后，分派子线程去处理这些请求，那么主线程就可以继续接收用户的下一个请求。但是子线程开始之后就会与主线程失去同步，主线程无法知道子线程何时完成任务，因此JavaScript使用回调函数来处理异步任务的返回结果。在子线程开始执行时，主线程就不必再关心子线程执行的具体任务情况，直到回调函数执行并返回给主线程数据时，主线程才和子线程合并。

### 2、Ajax 

Ajax用于描述异步的网络请求，具体的实现是XMLHttpRequest 对象，它用于在后台与服务器交换数据，发出HTTP请求，并接收返回数据。在不重新加载整个网页的情况下，对网页的部分内容进行更新。

1. 创建对象：`var httpRequest = new XMLHttpRequest();`
2. 打开请求：`httpRequest.open(String method , String url , bool async)`
3. （可选）添加http请求头：`httpRequest.setRequestHeader(header, value)`
4. 发送请求：`httpRequest.send(data)` 其中参数data是http请求体内容，仅用于post请求。
5. 获取请求状态：`httpRequest.onreadystatechange`函数
	1. `httpRequest.status`属性：0: 请求未初始化 1: 服务器连接已建立 2: 请求已接收 3: 请求处理中 4: 请求已完成，且响应已就绪
	2. `httpRequest.readyState`属性： 200: "OK" 404: 未找到页面
6. 根据请求状态获取响应：
	1. 获取字符串形式的响应数据：`httpRequest.responseText`
	2. 获取XML形式的响应数据：`httpRequest.responseXML`

### 3、Promise 

Promise对象是一个用于处理复杂异步请求的对象，比如说某个任务需要连续的异步请求，上一个请求的返回结果作为下一个异步请求的参数。我们就可以使用Promise对象。

Promise异步操作一共有三种状态：进行中、已成功、已失败。对应英文是 pending, fulfilled, rejected。对于状态的改变，只有从进行中变为已成功或者是进行中变为已失败，只要状态变为成功或失败，那么状态就不会再改变，称为已定型 resolved。一旦新建Promise对象，就会立即执行无法取消，我们需要设置回调函数才能处理Promise内部发生的错误，当处于进行中的状态时，无法得知任务当前已经进展到哪一个阶段了。

我们可以通过构造函数来创建一个Promise对象。Promise的构造函数中只有一个参数，是一个函数，这个函数在构造之后会直接异步执行，这个函数又包含两个函数参数resolve和reject（回调函数），分别表示在接下来的步骤中成功时需要调用的函数和出现异常时需要调用的函数。resolve和reject函数可传入一个数据对象，resolve传入这个参数在下一步骤中会做为下一个步骤的函数参数传入。reject传入的这个参数最后会当做异常步骤中的函数的参数。

```js
/*
说明：promise的构造函数
参数：resolve和reject回调函数，正常情况下调用resolve，出现异常时调用reject
	 resolve 和 reject 的作用域只有起始函数，不包括 then 以及其他序列；
	 resolve 和 reject 并不能够使起始函数停止运行，还需要 return。
返回值：传递给下一个异步请求的数据
*/
var promise = new Promise(function(resolve, reject){
    if(一切顺利){
        resolve('这里传入给下一次异步请求传递的信息')
	}else{
        reject('这里传入发生异常时你想传递的信息')
    }
})
/*
说明：promise的then方法可以有多个，promise会依照代码顺序依次执行这些代码
参数：是上一个异步请求的返回值传入的数据
返回值：传递给下一个异步请求的数据
*/
promise.then(function(value){	
    return value
})
/*
说明：promise的catch方法处理执行异步请求的异常情况
参数：reject函数的参数或者是then方法throw抛出的数据
*/
promise.catch(function(error){
    //处理异常
})
/*
说明：promise的finally方法无论是否出现异常都会执行
*/
promise.finally(function(){
    //执行善后工作
})
```

### 4、async 异步函数

* async：是用于修饰异步操作的关键字，用async修饰的函数就叫做async函数。async函数会返回一个Promise对象，我们可以对该对象进行与Promise相关的操作。
* await：在函数体内部，我们可以使用await 操作符，用于等待一个Promise对象的处理结果，在等待期间函数会停留在该语句上，直到Promise对象返回结果。

```js
Promise 
promise = async function p(params){
	result = await Promise对象
    return "p"
}
```

### 5、Generator 函数

Generator 函数的构建：使用function* ，内部使用yield表达式。在定义函数之后并不会立即执行，需要调用遍历器对象的next方法，就会分步的执行函数内部的一个代码段。next方法如果传递参数的话会作为上一个yield表达式的返回值，从而向函数体内部的某个地方传递数据。可以改变下一步函数的操作。return方法会返回给定的值，并结束Generator 函数。

### 6、Axios

Axios对象是在 promise 对象基础上封装的一个用于异步请求的JavaScript对象，该对象的使用流程如下：

1. 安装并导入Axios对象
2. 创建请求并配置请求参数
3. 发送异步请求
4. 获取请求数据

我们可以在发送异步请求之前让Axios对象知道我们要请求的URL、请求方法、请求头信息等一系列HTTP请求需要的数据，这就叫做**请求配置**。配置方法有两种：

1. `axios(Object config) `：传入一个对象，对象里面填写配置信息。
2. `axios(String url [, Object config])`：只传入一个URL，其余配置信息为可选。

下面说明一下详细的配置信息情况

| 属性             | 说明                                                         | 其他 |
| ---------------- | ------------------------------------------------------------ | ---- |
| url              |                                                              | 必需 |
| method           |                                                              |      |
| baseURL          |                                                              |      |
| transformRequest | 允许在向服务器发送前，修改请求数据。只能用在 'PUT', 'POST' 和 'PATCH' 这几个请求方法。后面数组中的函数必须返回一个字符串，或 ArrayBuffer，或 Stream |      |
|                  |                                                              |      |

如果没有太多请求参数要填写，我们也可以直接通过HTTP请求方法名称来发送请求。axios为每种请求方法都提供了一个别名方法，简化异步请求的步骤，这种情况下，就无需进行请求配置，直接调用别名方法来发送请求。

* axios.request(config)
* axios.get(url[, config])
* axios.delete(url[, config])
* axios.post(url[, data[, config]])
* axios.put(url[, data[, config]])

响应结构

| 属性 | 说明 |
| ---- | ---- |
|      |      |



## 六、TypeScript

