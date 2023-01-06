# Vue 练习

## 一、模板

vue需要基于HTML的模板操作HTML 的 DOM 对象。模板的语法有插值和指令两种。

### 1、插值

#### 1）对文本进行操作

使用双大括号文本插值，可以将插值处的文本和变量绑定。当绑定数据对象的值发生改变时，插值的内容会发生更新。通过使用`v-once`指令，插值将只执行一次。

```html
<span v-once>插值语法{{msg}}</span>
```

 如果意图是插入HTML代码的话，则需要使用v-html指令

```html
<p>
    使用v-html来插入可被浏览器解析的HTML代码<span v-html="rawHtml"></span>，rawHtml指的是变量名
</p>
```

#### 2）对HTML标签属性进行操作

使用双大括号不能作用在HTML的标签属性上，在这种情况下需要使用`v-bind:`指令绑定到具体的属性前面

```html
<div v-bind:id="myid">
    使用v-bind绑定到id属性上，其中myid是变量名
</div>
```

**注：**对于所有的数据绑定，都支持使用JavaScript的表达式，但不能写JavaScript语句。

### 2、指令

指令是带有v-前缀的HTML属性，当指令所对应属性的值发生改变时，将会对这个DOM元素产生影响

```html
<p v-if="seen">
    如果seen这个变量是true，则能看到这个文本。如果为false则P这个元素会被移除
</p>
```

#### 1）带参数指令

一些指令能够接受一个参数，在指令名称后以冒号表示，参数可以是DOM元素的属性，也可以是一个事件或者是一个JavaScript函数。

```html
<a v-bind:href="url">href是参数，告知v-bind将该元素的href属性和表达式url的值绑定</a>
```

#### 2）动态参数

也可以使用方括号括起来的表达式作为指令的参数，如上指令可修改为：那么url和attribute都是变量，需要传递进来值

```html
<a v-bind:[attribute]="href">...</a>
```

**注：**对于`v-bind` 和 `v-on`两个指令，分别可用缩写`:` 和`@` 代替

## 二、计算属性和侦听器

### 1、计算属性

在模板中放入太多的逻辑会让模板太复杂并且难以维护。对于任何复杂的逻辑，都应该使用计算属性。

```html
<div id="example">
    <p>
        使用基本属性的插值语法{{ msg }}
    </p>
    <p>
        使用计算属性{{ computedMsg }}
    </p>
</div>
```

```javascript
//实际上不需要定义新的对象，直接放在Vue的参数中。这样写并不好，只是为了初学时容易看清结构
var for_vm={
    el:"#example",
    data:{				//data为插值语法所用
        msg:"hello"		
    },
    computed:{			//computed为计算属性所用
		computedMsg: function(){
            return this.msg.split("").reverse().join("")
        }
    }
}
var vm=new Vue(for_vm)
```

计算属性只有它依赖的数据发生改变后才会重新求值，也就是说只要源数据不发生改变多次访问computedMsg不会再次执行函数，而是返回之前的计算结果。对于vue中的methods，每次调用都会执行相应的函数。计算属性不只是可以获取值，也可以设置值。即getter和setter。

### 2）监听器

vue通过watch选项提供了一个更通用的方法去响应数据的变化，当需要在数据变化时执行异步操作或者执行开销比较大的操作时，使用这种方式。





## 三、class和style的绑定

如果要操作元素的class列表和HTML内联样式时，vue做了专门的增强。

### 1、绑定HTML的class

#### 1）对象语法

* 我们可以传给v-bind:class一个对象，通过对象属性值(boolean类型)的真假，来动态的切换该class是否存在。
* 这个指令也可以和普通的class属性并存。
* 对于绑定的对象，除了定义在内联属性中。也可以定义在vue对象的data属性中。例子如下：

```html
<div class="static" v-bind:class="{active : isActive, 'text' : isTrue}"> </div>
```

#### 2）数组写法

我们可以把一个数组传给v-bind:class，应用这些class列表。

```html
<div v-bind:class="[activeClass, errorClass]"> </div>
<script>
    data:{
        activeClass:'active',
        errorClass:'text'
	}
</script>
```

### 2、绑定内联样式

与绑定class的用法类似，只不过是v-bind:style，我们可以直接在里面写上对象名称（变量名）然后在vue的data中定义它。

```html
<div v-bind:style="styleObject"></div>					<!--对象写法-->
<div v-bind:style="[styleObject, styleObject2]"></div>	  <!--数组写法，将多个样式对象应用在同一个元素上-->
<script>
	data:{
        styleObject:{
            color:'red'
        }
    }
</script>
```

## 四、条件渲染

### 1、v-if

用于条件性的渲染一块内容，只有在指令表达式值（boolean类型）返回true的时候才会被渲染。可以和v-else，v-else-if结合使用。

```html
<div v-if="isTrue1">渲染我-1</div>
<div v-else-if="isTrue2">渲染我-2</div>
<div v-else="isTrue3">渲染我-3</div>
```

使用`<template>`标签可以用来渲染分组。

### 2、v-show

与v-if用法大致一样。但是它只是简单的切换元素是否可见。

### 3、v-if 和 v-show

## 五、列表渲染

### 1、v-for

当vue更新用v-for渲染的列表时，为了给vue一个提示，一遍他能够跟踪每个元素的身份，从而重新排序或者重用该元素，我们需要给每一项提供一个唯一的key属性。如数组用法中的v-bind:key。

* 数组用法：用v-for指令基于一个数组来渲染列表，需要使用item in items形式的语法，遍历参数可选下标 item, index(option)

```html
<ul id='example'>
    <li v-for='item in items' v-bind:key='item.message'>{{item.message}}</li>
</ul>
<script>
	var vm=new Vue({
        el:'#example',
        data:{
            item:[
                {message:'1'},
                {message:'2'}
            ]
        }
    })
</script>
```

* 对象用法：可以用v-for来遍历一个对象的属性值部分，遍历参数可选 value, name(option), index(option) in object

```html
<ul id='example'>
    <li v-for="value in object">{{value}}</li>
</ul>
<script>
	new Vue({
        el:"#example",
        data:{
            object:{
                title:'如何学习Vue',
                author:'武沛鑫'
            }
        }
    })
</script>
```

* v-for可以接收计算属性，我们可以使用计算属性来呈现过滤后，或者排序后的列表。
* v-for也可以接受整数。
* 可以在`<template>`上使用v-for以便渲染包含多个HTML元素的内容。
* 在组件中使用v-for，key是必须的。如果想要把数据传递到组件内，我们需要使用prop属性。

## 六、事件处理

### 1、用法

使用`v-on`指令监听DOM事件，并在出发时运行一些JavaScript代码。事件处理的方法有三种：

1. 在事件的值处我们即可以写变量名（存储在data中）
2. 也可以写函数名（存储在methods)
3. 也可以直接作为JavaScript的表达式

```html
<div id="example">
    <button v-on:click="count += 1">增加按钮1</button><!--作为表达式-->
    <p>按钮被按下了{{count}}次</p>
    
    <button v-on:click="hit">增加按钮2</button>><!--写函数名-->
</div>
```

```javascript
var vm=new Vue({
	el:"#example",
    data:{
		counter:0
    },
    method:{
		hit:function(){
			cosole.log("按下了按钮")
        }
    }
})
```

### 2、修饰符

#### 1）通用修饰符

* .stop：阻止单击事件继续传播
* .prevent：提交事件不再重载页面
* .capture：使用事件捕获，内部元素触发的会先在此处理，然后在交给内部元素
* .self：只针对与当前元素，而不针对其内部元素
* .once：事件只触发一次
* .passive：被动事件

注：修饰符可以串联，实现效果的叠加。串联的时候顺序的不同会产生不同的最终效果。

#### 2）按键修饰符

* .enter：只有按键（键盘上的键）是enter键时才会触发事件
* .tab
* .delete
* .esc
* .space
* .up
* .down
* .left
* .right

## 七、表单输入绑定

### 1、用法

使用v-model在表单元素上创建双向数据绑定，v-model对于不同的HTML元素使用了原始HTML的不同属性。并抛出不同的事件。

* text textarea元素使用value属性和Input事件
* checkedbox radio元素使用checked属性和change事件
* select字段将value作为属性和change事件

```html
        <div id="root">
            <input v-model="message1" placeholder="一个文本框">
            <p>第一个文本框输入的文本是:{{message1}}</p>
            <br>

            <textarea v-model="message2" placeholder="多行文本"></textarea>
            <p style="white-space: pre-line;">第二个文本框输入的文本是:{{message2}}</p>
            <br>

            <input v-model="checked" type="checkbox" id="checkedbox">
            <label for="checkedbox">你的选择状态:{{checked}}</label>
            <br>

            <input type="checkbox" id="草莓" value="草莓" v-model="checkedlist">
            <label for="草莓">草莓</label>
            <input type="checkbox" id="菠萝" value="菠萝" v-model="checkedlist">
            <label for="菠萝">菠萝</label>
            <br>
            <span>你选择了{{checkedlist}}</span>
        </div>

        <script type="text/javascript" >
            const vm =new Vue({
                el:'#root', //指定当前Vue实例为哪个容器服务 and find element which id is root(css selector)
                data:{      //该data存储数据，供el对应容器使用，其使用时对应文本需要替换为 {{键}}
                    message1:"",
                    message2:"",
                    checked:false,
                    checkedlist:[]
                }
            })
        </script>
```

### 2、修饰符

* .lazy：默认情况下，双向数据绑定是在每次输入一个字符时进行同步，但是可以添加lazy修饰符，那么只有在输入完毕之后（change事件发生后）才会进行同步。
* .number：通常情况下用户输入的都是字符，number修饰符将用户输入的数字字符转换为数字。
* .trim：用来过滤用户输入首尾的空白字符

## 八、组件

### 1）组件的使用

#### 组件的概念

组件是可复用的Vue实例，每个组件都带有一个名字。我们可以通过new Vue创建的根实例中，把这个组件作为自定义的元素使用。因为组件是可复用的Vue实例，所以他们与Vue接收相同的选项，如data, computed, watch, methods 以及生命周期钩子等。但是不能接收el这样根实例特有的选项。我们也可以复用组件，对于每次组件的复用，都会有一个新的实例被创建。组件中需要HTML模板，我们把这个模板放在组件的template属性中。注意每个组件必须有一个根元素，将模板的内容包裹在一个父元素中解决这个问题。

在vue实例中，data可以是一个对象。但是在组件中，data必须是一个函数，返回一个对象。因此每个实例可以维护一份独立的对象拷贝，即使组件复用中的其他组件的数据发生改变，该组件的数据也不会发生变化。

#### 创建组件

创建一个组件的语法如下：

``` javascript
Vue.component("component_a", {...})//全局注册
var component_b={...}			 //局部注册
new Vue({
    el:"...",
    component:{
        component_a,				//全局注册的组件随时使用
        "component_b":component_b	 //局部注册的组件只能在当前文件作用域下使用。
    }       
})
```

#### 组件注册

在注册组件的时候，我们需要给该组件一个名字，作为识别这个组件身份的标识。组件名就是中的第一个参数。组件名称最好使用多个单词驼峰命名法，即每个单词首字母大写。

组件注册分为全局注册和局部注册，上述语法就是全局注册的组件。全局注册的组件可以用在任何一个新创建的vue根实例的模板中。而有些时候，我们只是想在某个小范围使用这个组件，那就把他注册为局部组件。这种情况下，我们使用一个普通的JavaScript对象来定义这个组件的内容，然后在vue实例中定义你想要使用的局部组件。局部注册的组件在其子组件中不可用。

在模块系统中注册局部组件，我们在项目目录中创建一个中component目录，并且将每个组件放在一个单文件组件中。在局部注册一个组件之前，我们需要先导入它，然后再开始使用。

```javascript
import ComponentA from './ComponentA'

export default {
	components:{
        ComponentA
    }
}
```

#### 组件的使用

```javascript
Vue.component("myBotton",{
	data : function(){
		return {count : 0}
	},
	template:'<button v-on:click="count++">你按了{{count}}次</button>'
})
new Vue({
	el:'#example'
})
```

```html
<div id="example">
    <!--将组件名称作为自定义的元素使用-->
    <mybutton></mybutton> 
    <mybutton></mybutton>
</div>
```

### 2）向组件传递元素

#### 通过prop传递元素

通过prop来向组件传递元素，prop是你在组件注册时可选的一个自定义属性。当一个值传递给prop的值时，这个值就变成了组件实例的属性。

子组件中prop的第一种写法：字符串数组形式。这个时候，数组中的每个字符串对应的都是一个变量，我们可以在HTML模板中使用插值语法来获取变量对应的值。

下面也展示了在父组件中如何向子组件传递prop。第一种是传递静态的prop；第二种是传递动态prop，在父组件的HTML模板中，对于子组件标签，使用v-bind绑定父组件中的data数据。该数据就传递进了子组件中。如果使用v-bind，那么之后双引号内部的就默认为变量名（实际上是JavaScript表达式），而不是单纯的静态字符串。这个变量它的值既可以是值、数组、对象。

```html
<div id="example">
    <blog title="this is not variable,it just a string"></blog>						<!--父组件中传递静态prop-->
	<blog v-for="post in posts" v-bind:key="post.id" v-bind:title="post.title"></blog><!--父组件中传递动态prop-->
</div>
```

```js
Vue.component("blog",{
    props:['title'],
    template: `
    <div class="blog">
    	<h2>{{title}}</h2>
    	<div v-html="post.content"></div>
    </div>
    `
})
new Vue({
    el="#example",
    data:[
    	{id:1,title:"标题1"},
        {id:2,title:"标题2"},
        {id:3,title:"标题3"}
    ]
})
```

子组件中prop的第二种写法：对象形式。JavaScript变量的类型是没有显式的声明出来的。会给我们具体编程产生一些不方便的地方。把prop写成对象这种情况是因为我们想要规定prop中的每一个变量需要有一个指定的类型。那就为类型检查提供了方便。

```js
prop:{
    title:String,			//规定title是一个string类型的变量
    time:[String,Number],	 //规定time可能是多个数据类型
    author:{				//规定author是一个object类型的变量，且必须填写
        type: Object,
    	required: true
    },
    price:{					//规定price是带有默认值的数字，主要default不要和required同时使用
        type:Number,
        default:100
    },
    comment:{				//规定comment是带有默认值的对象，且对象或数组的默认值必须是从一个函数中获取
		type:Object,
         default: function(){
         	 return {message : 'hello'}      
         }
    },
    isError:{				//自定义验证函数，值必须匹配以下数组中的字符串
        validator:function(value){
            return ['success','warning','danger'].indexOf(value)!=-1
        }
    }
}
```

#### 注意：单向数据流

父组件向子组件传递的数据（在prop中）是单向流动的，也就是从父级的prop流到了子级的prop而不能反向流动。那么每当父组件发生变更时，子组件的所有prop中存放的数据都会相应的刷新。因此我们不应该在子组件中更改prop的值，只在父组件中修改，子组件负责展示。如果想要在子组件中对数据的呈现方式做出改变，也就是修改数据格式，我们可以使用计算属性。这时，我们更改的只是数据的展现方式。

#### 注意：父组件向子组件中传入未在prop中定义的属性

因为在编写子组件的时候，有时我们可能无法确定父组件会向子组件传递什么数据。需要父组件决定传入的数据。一个子组件实际上是可以接受父组件传来的任意属性名的属性。而不一定非得在prop中定义，这样的属性会被添加到这个子组件的根元素上。传入的这个属性可能会替换或者合并子组件根元素上的已有属性。对于绝大多数属性来说，从父组件提供给这个子组件的属性值会替换掉原来的值，但是class和style不一样，从外部传进来的值会和原来已有的值合并，造成效果的叠加。

### 3）监听子组件事件

1. 父组件向子组件单向的传递数据，我们可以通过在vue实例data属性中新建一个数据项，然后子组件就可以利用该数据。
2. 子组件向父组件传递数据，我们可以使用一个自定义事件。父组件就可以通过v-on监听子组件实例的事件传输数据，子组件使用内建的$emit方法。传入事件名称，触发一个事件。
3. 对于自定义的事件，建议命名方式为：不同单词之间加上横线，因为v-on事件监听器在HTML模板中会被自动转换为全小写，那么驼峰命名法的事件名就不会起作用。

### 4）通过插槽分发内容

#### 插槽的概念

Vue实现了一套内容分发的API，使用`<slot>`元素作为分发内容的出口。slot就是插槽的含义。

我们可以在父组件的HTML模板中，在引用子组件的时候，子组件元素中间（起始标签和结束标签之间）可以插上任意一段文字、模板代码、HTML、甚至是其他组件。在下面的这个例子中 “`<slot></slot>`” 会被自动替换为 “插入的数据” 这几个字。这也就是插槽这个词语的含义。即父组件可以在它的HTML模板中指定任意一段文档，这样插入到子组件的相应位置中。

当然，如果子组件模板中没有定义`<slot>`这个元素的话，父组件传来的文档就不知道该插入到哪个位置，那么这个文档就会被子组件丢弃不起作用。

```html
<!--子组件my_component的模板中-->
<a v-bind:href="url">
	<slot>当父组件没有插入数据时会显示该信息</slot>
</a>

<!--父组件的HTML模板中-->
<my_component url="/profile">插入的数据</my_component>
```

#### 插槽中变量的作用域

如果你想在插槽中使用变量（JavaScript表达式），那么这个变量必须是在父节点中定义的，父级模板里的所有内容都是在父级作用域中编译的；子模板里的所有内容都是在子作用域中编译的。所以说上述例子中如果插槽中想要访问my_component的url变量，是访问不到的，因为url是在my_component这个组件内部定义的。

#### 插槽中的备选内容

在父组件没有向子组件传递插槽中的信息时，子组件的插槽设置备选的内容也是很有用的。这个备用的内容只有在没有提供内容的时候才会被渲染。

#### 带有名称的插槽

如果我们有时候想要在子组件中的多个地方插入东西，那就需要多个插槽。这种情况下每个插槽需要有自己的名字才能够分别不同的插槽。

在子组件中，`<slot>`元素有一个属性name，可以用来定义不同的插槽。

在父组件中，我们在使用子组件的地方，还需要再额外添加一个`<template>`元素，并且在`<template>`元素上使用v-slot指令提供插槽的名称，在`<template>`元素内部提供该插槽的内容。没有使用`<template>`元素包裹的文档都会插入到默认的插槽中，一个不带name属性的插槽为默认的插槽，隐含的名字为default。

```html
<!--在子组件my_component的模板中写法-->
<a v-bind:href="url">
	<slot name="slot1">第一个插槽：当父组件没有插入数据时会显示该信息</slot>
    <slot name="slot2">第二个插槽：当父组件没有插入数据时会显示该信息</slot>
    <slot>默认插槽：当父组件没有插入数据时会显示该信息</slot>
</a>

<!--在父组件模板中的写法-->
<my_component url="/profile">
	<template v-slot:slot1>我应该插入到第一个插槽中</template>
    <template v-slot:slot2>我应该插入到第二个插槽中</template>
    我应该插入到默认的插槽中。
</my_component>
```

#### 扩展插槽的作用域

前面说到了，插槽中的变量的作用域是在父组件中的，但是有时候我们想让插槽也能够访问子组件的数据。那么就需要扩展插槽的作用域到子组件上，使父组件的插槽能够访问到子组件中的一些变量。因此我们需要在子组件和父组件对应位置做一下绑定。

在子组件中，我们在`<slot>`元素上使用v-bind绑定子组件的一个属性，假设子组件存在user属性。绑定在插槽上的属性称为插槽prop。

在父组件中，在`<template>`元素上使用带值的v-slot指令定义绑定在`<slot>`元素上的属性们的名称。这里把这些属性命名为slotPlops。

```html
<!--在子组件my_component的模板中写法-->
<a v-bind:href="url">
	<slot name="slot1" v-bind:user="user">第一个插槽：{{user}}</slot>
    <slot v-bind:user="user">默认插槽：{{user}}}</slot>
</a>

<!--在父组件模板中的写法-->
<my_component url="/profile">
	<template v-slot:slot1="slotProps">我应该插入到第一个插槽中:{{slotProps.user}}</template>
    <!--对于默认插槽可以简写-->
    <template v-slot="slotProps">我应该插入到第二个插槽中：{{slopPlops}}</template>
</my_component>
```

在ES6语法的支持下，我们可以使用解构赋值来简化变量的书写。因为实际上插槽的内容是包括在一个函数里的。

**注：**`v-slot:`的缩写为#

### 5）动态组件

#### 动态组件介绍

如果有好几个组件都被放在一个区域，我们希望通过鼠标点击在不同组件之间动态的切换，那么我们需要通过Vue提供的`<component>`元素。在`<component>`元素中使用v-bind绑定一个名叫is的属性，通过属性值的切换来动态的切换组件。当这个值改变为对应组件的名称后，该位置就会切换到相应的组件上。

```html
<div id="example">
    <component v-bind:is="currentTabComponent"></component>
</div>
<script>
      Vue.component("组件1", {...});
      Vue.component("组件2", {...});
      new Vue({
        el: "#example",
        //通过v-model动态的修改当前组件名称（即变量currentTab）
        //vue就会检测到currentTabComponent发生的变化。从而切换组件。
        //currentTab也可以直接是一个组件对象，即Vue.component("组件1", {...})中的{...}
        data: {currentTab: "组件1"},
        computed: {
          currentTabComponent: function() {
            return this.currentTab;
          }
        }
      });
</script>
```

#### 保持动态组件的状态

在这些动态组件之间进行切换的时候，切换走的组件会被Vue销毁，切换进的组件会被Vue创建出来，我们有时候想要保持这些组件的状态，不想让他反复的创建和销毁。那么我们就可以使用一个`<keep-alive>`元素，将这个动态组件包裹起来。

#### 异步组件

**待解决**

#### 边界情况

**待解决**

## 九、动画效果

### 进入离开

### 列表过渡

### 状态过渡



## 十、复用代码

### 1、混入

混入的意思就是把一些组件通用的属性和方法单独的抽离出来，作为这些组件公共的方法。在使用的时候，把这个对象用混入的方式插入进组件中。那么该组件就会拥有这个对象。

```js
var say={		//定义一个好几个组件都会使用的方法
    methods:{
        hello(){
            console.log("hello, my friends!")
        }
    }
}
new Component({	//将这个对象混入进一个需要它的组件中
    mixin:[say]
})
```

需要注意的是：

* 当组件中原有的东西与混入的对象产生冲突的时候，比如说有重名的属性，方法。这些冲突会以恰当的形式合并。不能合并的，实际的值以组件内的为准。
* 对于生命周期钩子函数的混入，他们的代码内容会合并，并且混入对象钩子的代码会在组件自身钩子代码之前被调用。
* 对于一些对象属性，比如methods, components, directives将会被合并为同一个对象。对象内部的键名冲突时，取组件内部的键值对。
* 混入也可以进行全局混入，这样做的话，它将会影响之后创建的每一个Vue实例。
* 我们也可以自定义合并的策略，是简单的覆盖已有的值，还是想要按照某种逻辑合并。

### 2、自定义指令

如果没有指令能够实现我们想要的功能，那么我们也可以自定义指令。语法如下

```js
//全局注册指令
Vue.directive('directive_name',{
    inserted:function(el){
        el.focus()
    }
})
//局部注册指令，在某个组件内部
directives:{
    focus:{
        inserted:function(el){
            el.focus()
        }
    }
}
```

一个指令对象可以提供如下的几个函数：

* bind：绑定时，当指令第一次绑定到元素上时会调用，只调用一次，可以做一些初始化设置。
* inserted：？？？被绑定的元素插入父节点时会调用
* update：所有组件的VNode更新的时候会调用。
* componentUpdated：指令所有组件的VNode及子VNode全部更新时调用
* unbind：指令与元素解绑时调用

函数可以传入以下参数：

* el
* binding
	* name
	* value
	* oldValue
	* expression
	* arg
	* modifiers
* vnode
* oldVnode

### 3、渲染函数

渲染函数的使用有一定的背景。一般情况下，我们使用HTML模板来创建网页，这比直接写原生的HTML代码要好，可以减少很多重复的操作，但是当HTML模板也无法满足我们的要求，会产生许多重复的操作时，我们就需要借助JavaScript代码用一个函数来写HTML模板。这就是渲染函数，指的是可以创建HTML模板、HTML代码的函数。

首先，HTML的每个DOM元素，都可以看做是一个节点node，这些节点彼此之间是一种树形结构，vue通过建立一个虚拟的DOM节点的描述来追踪自己要如何改变真实的DOM元素，我们把Vue虚拟出来的节点描述称为Virtual Node(VNode)，虚拟DOM是我们对Vue组件树建立起来的整个VNode的称呼。

在一个渲染函数render中，它会传入一个回调函数createElement，这个函数负责创建虚拟DOM。下面描述了createElement接收的参数类型：

```js
createElement(
	{String | Object | Function},		//必填
    {Object},						   //可选
    {String | Array}					//可选
)
    
```

每个参数的解释如下：

* 第一个参数：HTML标签名、组件选项对象、anync函数
* 第二个参数：与模板中属性对应的数据对象
* 第三个参数：子级虚拟节点

// TODO

### 4、插件

插件用于为Vue添加额外的功能。我们通过全局方法Vue.use(plugin_name)使用插件，他必须在创建Vue实例对象之前完成。Vue.use方法会自动阻止多次向Vue实例中注册相同的插件。因此多次调用也只会注册一次。

### 5、过滤器

过滤器尝尝用于文本格式化，通过过滤器函数，我们可以将输入的数据转变成另一种格式输出。过滤器用在双括号`{{}}` 插值和v-bind的表达式中。过滤器需要添加在JavaScript表达式的尾部，使用管道符号 `|` 表示。

* 我们可以定义局部组件过滤器，也可以定义全局过滤器，当全局过滤器和局部过滤器重名时，会采用局部过滤器。
* 过滤器在使用的时候可以串联
* 过滤器也可以接收多个参数，默认是它管道符号 `|` 之前传递来的数据。

第一步：在JavaScript的vue组件中定义过滤器

```js
filters:{
    formatId:function(value){
        //填入处理逻辑
        return value
    }
}
```

第二步：在HTML模板中使用过滤器

```html
<!-- 在双花括号中，串联的过滤器 -->
{{ Id | formatId | reformatId }}

<!-- 在 v-bind 中 -->
<div v-bind:id="Id | formatId"></div>
```

## 十一、路由

### 1、路由匹配

#### 使用`<router-link>` 创建 a 标签来定义导航链接

* HTML模板中需要的配置：使用`<router-link to="/">`标签来导航，引导用户点击来显示该路由组件。需要注意的是，该路由实际上是`<a>`标签，超链接。然后在合适的地方使用`<router-view>`来展示该路由匹配到的组件。
* 在JavaScript中，如果我们针对路由信息专门写了一个js文件，在这个文件中我们需要进行如下的操作。
	* 定义路由组件。可以从其他文件中使用ES6语法import进来。
	* 定义路由。每一个路由都应该映射到一个组件。
	* 创建router实例，然后传进去刚刚定义的路由。
	* 在App.js中创建并挂载根实例，并将路由实例router注入进来
* 在全局的APP注入路由之后，我们就可以在任何组件中通过`this.$router`访问路由器，或者通过`this.$route`访问路由
* 命名路由：有时候路由路径过长，我们希望通过名称来表示一个路由，而不是通过路径。那么我们可以使用命名路由。在router的js配置里，我们需要给对应的路由添加name属性。在HTML模板中，我们需要使用v-bind绑定to属性，那么对应的值也就变成了JavaScript表达式，我们就可以给表达式传递一个对象。这样可以通过路由传递更多的信息。
* 命名视图：有时候我们想要同时展示多个视图，之前的视图一次只能展示一个，它无法对不同的视图同时的展示，那么就需要给每个视图命名。如果一个路由视图没有名字，它默认名字为default。对于命名的路由视图，同一个路由就需要多个组件，多个组件在路由配置中就需要使用components，并定义对象

```html
<p>
    <router-link v-bind:to="foo">Go to Foo</router-link><!--这里foo是一个JavaScript表达式-->
    <router-link to="/bar">Go to Bar</router-link>
</p>
<!-- 路由匹配到的组件将渲染在路由视图这里 -->
<router-view></router-view>							<!--默认路由视图-->
<router-view name='router_1'></router-view>			  <!--命名路由视图-->
<router-view name='router_2'></router-view>			  <!--命名路由视图-->
```

```js
// 定义路由
const routes = [
  { path: '/foo', name:'foo', component: Foo },
  { path: '/bar', component: Bar },
  { path: '/router',components:{
      default:Foo,
      router_1:Bar,
      router_2:Baz
  }}
]

// 创建 router 实例，然后传 `routes` 配置
const router = new VueRouter({
  routes // (缩写) 相当于 routes: routes
})
const app = new Vue({
  router
}).$mount('#app')
```

* 动态路由匹配：有时候我们需要将某种模式匹配到的所有路由都映射到同一种组件，这时候就需要动态路由匹配来达到这个效果。如在一个路径后面使用 `/:`标记，当匹配到对应路径是，之后的路径就会当做参数被设置到`this.$route.params`，可以在每个组件内使用。这些匹配到的参数叫做路径参数。

| 模式                          | 匹配路径            | $route.params                   |
| ----------------------------- | ------------------- | ------------------------------- |
| /user/:username               | /user/evan          | {username:'evan'}               |
| /user/:username/post/:post_id | /user/evan/post/123 | {username:'evan',post_id='123'} |

* 那么在使用路径参数是，原来的组件实例将会被复用。如果我们想要对路径参数的变化做出响应的话，可以使用watch来监测$route对象的变化。或者引入路由守卫。
* 如果要匹配未被已定义的路由匹配到的路径，那么我们可以使用通配符来匹配任意路径，这时候含有通配符的路径应该放在最后，通常用于客户端404错误。实际上，路由匹配的优先级是跟代码顺序相关的，路由定义的越早，匹配的优先级就越高。

#### 编程式定义导航

之前是使用`<router-link>` 创建 a 标签来定义导航链接，那这次可以借助router的实例方法，编写代码来实现定义导航。在vue实例内部，我们可以通过$router访问路由实例，因此调用this.$router.push()也可以导航到对应的URL。这两种定义导航的方式是等价的。

该方法的参数可以是一个字符串路径，也可以是描述地址的对象。通过这种方法会再浏览器的历史记录中留下痕迹。

```js
router.push('home')							//传入字符串路径
router.push({path:'home'})					//传入的是对象
router.push({name:user, params:{userId:'123'}})//传入的是命名的路由
```

调用this.$router.replace()跟this.$router.push()是类似的结果，但是他不会向浏览器的历史记录中留下痕迹。也就是说它会替换掉当前的历史记录。

调用this.$router.go(n)会在历史记录中向前或向后退n步。

### 2、嵌套路由

想要使用嵌套路由，需要在routers数组中的对应的路由对象使用children配置。

```js
const router=new VueRouter({
    routes:[
        {
            path:'/user',
            conponent:User,
            children:[
                {
                    path:'profile',
                    conponent:UserProfile
                },
                {
                    path:'',
                    conponent:UserPost
                }
            ]
        }
    ]
})
```

要注意的是，以/开头的嵌套路径被看作根路径，在嵌套组件中就不能在设置/，即无需在设置嵌套的路径了。如果访问到一个没有被匹配的路径的话，那就不会渲染任何东西，如果想要渲染点什么，需要准备一个空路由。如上所示。

### 3、重定向和别名

重定向：将一个原本路径为a的路由重定向到b，是在routes中的redirect属性上配置的。redirect属性的值可以为三种：

* 字符串。在单纯指定其路由路径的时候，使用字符串值。
* 对象。如果要指定一个命名的路由，使用对象来说明。
* 方法。通过一个方法，封装一些逻辑，动态的返回重定向目标。

别名：如果使用别名的话，用户访问 `/a` 的别名是 `/b`，意味着，当用户访问 `/b` 时，URL 会保持为 `/b`，但是路由匹配则为 `/a`，就像用户访问 `/a` 一样。

```js
const router=new VueRouter({
    route:[
        {path:'/a1', redirect: '/b1'},								 //通过字符串重定向
        {path:'/a2', redirect: {name: 'b2'}},						 //通过对象重定向
        {path:'/a3', redirect: to => { return stringPath/PathObject }},//通过方法重定向
        {path:'/a', component:A, alias: '/b'}						 //为a路径创建别名
    ]
})
```

### 6、路由组件传参

如果我们想要在组件内部获取导航到该组件的路由上面的参数，怎么办？通常情况下，组件本身有一个$route路由实例对象，我们可以通过他获取路由参数。但是我们不能轻易的在组件中使用$route，因为这样会让组件和它对应的路由形成高度的耦合，限制了组件的灵活性。

vue组件中props这个属性原本是用于父组件向子组件传递数据使用的，我们在路由中也可以使用它。主路由器通过props来向组件传递该路由的参数，那么我们就不需要通过调用组件实例对象的$route对象来获取路由参数。通过路径向组件传递参数总共有三种方法：

#### 1）布尔模式

当路由器中该路由设置props属性为true时，vue就会将在URL路径上的参数传递到组件的props中。注意到在组件的props中也需要键值与参数名相对应的属性。才能够传递成功。

#### 2）对象模式

当路由器中该路由设置的props属性设置为一个对象的时候，该对象的键值对信息就会静态的传递给组件的props中，注意在组件的props中也需要键值与参数名相对应的属性。才能够传递成功。

#### 3）函数模式

当路由器中该路由设置的props属性设置为函数的时候，我们可以自定义路由器中获取到的路由参数和组件的props的某个属性之间的映射关系。注意在组件的props中也需要键值与参数名相对应的属性。才能够传递成功。

```js
//在组件中
vue.component({
  props: ['id','name',],
})
//在主路由器中
const router = new VueRouter({
  routes: [
    { path: '/user/:id', component: User, props: true }
    { path: '/static', component: Hello, props: { name: 'world' }},
    { path: '/dynamic/:years', component: Hello, props: dynamicPropsFn }
  ]
})
function dynamicPropsFn (route) {
  const now = new Date()
  return {
    name: (now.getFullYear() + parseInt(route.params.years)) + '!'
  }
}
```

### 7、导航守卫

导航守卫分为全局前置守卫、全局解析守卫、全局后置守卫、路由独享守卫、组件内守卫。在用户输入URL跳转路由的时候，这几个路由守卫会依次触发。守卫异步执行，路由跳转在所有守卫处理完毕之前处于等待中。

每个守卫方法接收三个参数`function(to, from, next){ ... }`

* to：即将进入的路由对象
* from：当前导航正要离开的路由
* next：函数。在任何给定的导航守卫中都需要严格调用一次，否则路由就不会被解析
	* next()：进入下一个路由守卫，如果全部守卫已工作完成，则进入目标导航
	* next(false)：中断当前导航，URL地址会重置为之前离开了的地址
	* next('/') 或者 next({path : '/' })：跳转到另一个不同地址
	* next(error)：终止导航并报错。

完整的路由解析流程如下：

1. 路由导航被触发
2. 在失活的组件中调用`beforeRouteLeave`
3. 调用全局的`beforeEach`
4. 在重用的组件中调用`beforeRouteUpdate`
5. 在路由配置里调用`beforeEnter`
6. 解析异步路由组件
7. 在被激活的组件中调用`beforeRouteEnter`
8. 调用全局的`beforeResolve`
9. 导航被确认
10. 调用全局的`afterEach`
11. 触发DOM更新
12. 调用`beforeRouteEnter`守卫中传给next的回调函数，对于创建好的组件实例会作为回调函数的参数传入。

#### 1）全局前置守卫

使用`router.beforeEach(function(to, from, next){ ... })`注册一个全局前置守卫。

#### 2）全局解析守卫

在导航被确认之前，**同时在所有组件内守卫和异步路由组件被解析之后**，解析守卫就被调用。

#### 3）全局后置守卫

这个函数不能传入next函数，也不会改变导航本身

#### 4）路由独享守卫

对于某个特定的路由向，也可以为其单独定义路由导航使用beforeEnter属性

#### 5）组件内守卫

你可以在路由组件内直接定义以下路由导航守卫：

- `beforeRouteEnter`在进入该路由之前会被调用，此时不能获取组件实例的this
- `beforeRouteUpdate`在当前路由改变但是没有跳转离这个组件时，比如说传了一个跟之前不一样参数，组件会被复用
- `beforeRouteLeave`在导航离开该组件的时候，这时候通常用来禁止用户未保存修改之前离开。

### 8、路由元信息

首先，我们称呼routes配置中的每个路由对象为路由记录。如果我们想要在路由记录上添加一些我们自定义的信息，我们就可以在路由记录中添加meta属性。路由记录是可以嵌套的，一个路由匹配到的所有路由记录都会暴露给`$route.matched`数组，我们可以通过遍历$route.matched数组获取到meta字段。

### 9、过渡动效



### 10、数据获取

有时候在进入路由的时候我们需要从服务器获取数据，这时候我们可以通过两种方式来实现

* 导航完成之后获取：先完成导航，然后在组件生命周期内获取数据，在获取数据时显示加载中之类的提示
* 导航完成之前获取：在导航完成前，在路由守卫中获取数据，数据获取成功之后执行导航。

#### 1）导航完成之后获取

在组件创建完成后，我们可以在其生命周期的created中获取数据，此时vue绑定的数据还没有挂载到HTML页面上，这样我们就有机会在数据获取期间向用户展示一个“正在加载中”的状态。、

```html
<template>
	<div class='example'>
        <div v-if='loading'></div>
        <div v-if='error'></div>
        <div v-if='post'>
            <h2>{{post.title}}</h2>
            <p>{{post.body}}</p>
        </div>
    </div>
</template>
<script>
export default {
    data(){
        return {
            loading:false,
            post:null,
            error:null
        }
    },
    created(){
        this.fetchData()
    },
    watch:{//路由变化时，会再次执行该方法
        '$route':'fetchData'
    },
    methods:{
		fetchData(){
            this.error=this.post=null
            this.loading=true
            /*从服务器中请求数据。
            请求完成后设置this.loading=false
            返回成功时 this.post不为空，this.error为空，则只显示成功的消息。
            返回失败时 this.post为空，this.error不为空，则只显示失败的情况。
            */
            getPost(this.$route.params.id, (err, post) => {
                this.loading = false
                if (err) {
                  this.error = err.toString()
                } else {
                  this.post = post
                }
          })
        }
    }
}
</script>
```

#### 2）导航完成之前获取



### 11、滚动行为



### 12、路由懒加载



## 十二、vue-cli

### 1、vue-cli介绍

vue-cli即vue command line interface，是一个快速开发的脚手架，集成了许多前端开发中很好用的工具，并对这些内容进行了合理的默认配置，我们还可以通过插件来扩展vue脚手架的功能。

vue-cli由以下几个组件构成：

* @vue/cli：位置在全局安装，提供了vue命令，使用`vue help`可以查看cli所提供的命令
* @vue/cli-service：位置在项目的node-modules中，构建于webpack和webpack-dev-server上，提供了以下功能：
	* plugin：提供了加载其他cli插件功能
	* vue-cli-service：为项目内部提供了vue-cli-service命令
	* preset：预设了Webpack的配置，使我们无需在手动配置
* @vue/cli-plugin-name / vue-cli-plugin-name：这些是安装的插件包的名称，插件为我们提供一些额外的功能

###  2、预设preset

这个预设是我们在创建vue项目时，应该选择哪些插件，使用哪些前端开发的功能，preset预设配置会存放在安装路径下的的`.vuerc`文件中，我们可以直接编辑文件来指定我们在创建新项目时候都需要哪些东西。

### 3、插件plugin

我们可以为vue项目安装一些插件来增加一些额外的功能，注意到这跟npm install安装的软件包非常类似，而vue的这些插件大多数也来源于npm中为vue项目开发的公开软件。但是vue插件都是以@vue/cli-plugin开头的

### 4、服务cli-service

vue-cli-service为我们提供了一些命令，常用的命令如下，我们可以使用`vue-cli-service help command`查看命令的额外选项：

* `vue-cli-service serve`，类似于`npm run serve`启动一个开发环境下的前端服务器，我们可以通过参数指定端口，是否打开浏览器等选择。
* `vue-cli-service build`，类似于`npm run build`编译产生一个用于生产环境的软件包。
* `vue-cli-service inspect`，查看项目的Webpack配置。

### 8、配置项

针对vue脚手架的个性化配置有两种方法。主要讲解使用vue.config.js的情况。

1. 使用vue.config.js，它是一个可选的配置文件。如果存在该文件，会被vue脚手架自动加载。
2. 使用package.json中的vue字段，但是需要严格遵守JSON格式。

| 属性       | 说明 | 默认值 |
| ---------- | ---- | ------ |
| publicPath |      |        |
| outputDir  |      |        |
| assetsDir  |      |        |
| indexPath  |      |        |
|            |      |        |

## 十三、vueX

### 1、VueX介绍

当需要多个组件共享数据时，我们可以使用VueX。实际上是把多个组件共享的信息抽取出来，以全局单例模式进行管理。每个VueX应用核心都是一个仓库store，仓库是一个容器，包含着容器中大多数状态。但是VueX和单纯的全局对象也有所不同：

* VueX的数据存储是响应式的，当vue组件使用VueX时，如果仓库store中的数据状态发生变化，那么相应的组件也会得到更新。
* 不能直接改变store的状态，只有通过提交mutation，这样的话VueX就可以方便跟踪数据的变化。

注：mutation n 意为形式或结构的转变

### 2、生命周期

![vuex](https://v3.vuex.vuejs.org/vuex.png)

#### 1）state

首先每个应用程序仅包含一个store实例，其次存储在VueX（store）中的数据和vue实例中的data数据遵循相同的规则。

如何在vue组件中获取store的状态呢，换句话说如何获取store中的数据变化。回想起计算属性只有它依赖的数据发生改变后才会重新求值。最简单的办法就是在计算属性中返回某个数据的状态。我们需要vue使用VueX这个插件，然后将store注册进根实例中，那么我们就能在所有子组件中通过this.$store访问到。

```js
//程序入口js文件中
import store from "./store"
new Vue({
	el:"#app",
    store
})
//子组件中
computed:{
	count(){
        return this.$store.state.count
    }
}
```

如果一个组件需要从store中获取多个数据的状态时，将这些数据都声明为计算属性会有些重复冗余，我们可以使用`mapState`函数帮助我们生成计算属性。

```js
import { mapstate } from 'vuex'
export default {
	computed: {
		mapState({
			count: state => state.count,
        })
    }
}
```

#### 2）getter

我们可以把vuex中getter看做是vue组件的计算属性computed。而state相当于vue组件中的data。也就是说，如果多个组件需要用到全局的数据的时候，而用到这些全局数据时又需要一定的计算和过滤或者是其他的操作，那么使用getter是一个不错的选择。getter可以直接返回一个全局数据过滤后的数据。

getter内部数据对象的创建。getter会接收state作为它的第一个参数，我们可以在getter内部属性对应的方法中对state的数据进行进一步的操作。getter也可以接收其他getter作为其第二个参数，这是可选的。我们可以在其他getter操作的基础上进一步的操作。

```js
getter:{
	myData: (state) => {
		return state.todos.filter(todo=>todo.done)
	}
}
```

在组件中对于getter的访问，我们可以直接从store对象中的getter对象访问，即this.$store.getter

#### 3）mutations

更改VueX的store中数据的状态的唯一方法是提交mutation，mutations中的每一个mutation都有一个字符串的事件类型和对应的回调函数，即属性和属性对应的方法。我们在回调函数中进行状态的更改。回调函数接收state作为其第一个参数。mutation更像是一个事件，如果我们使用mutation改变数据的状态时。我们可以通过store的提交方法即store.commit()。这个方法的传参有两种方式，一种是简单参数，第一个参数是mutation事件名称，第二个参数是向mutation中提交的附件信息。第二种方法是直接传入一个对象，其中mutation事件名称应该在type属性中存放。

对于第一种方法，我们需要通过以下语法

```js
store.commit('your_mutation_name');
```

回调函数也可以再接收一个参数，这个参数是从调用它的那个组件中传递进来的。这个参数可以是一个简单的值，也可以是一个对象。我们称它为payload。如下。

```js
//在store中
mutation:{
	increment (state, payload){
		state.count += payload.amount
    }
}
//在某个组件中
store.commit('increment',{amount:10})
```

第二种通过传入对象提交事件的方法如下，此时提交的整个对象都将作为mutation事件的第二个参数payload，可以被mutation事件的回调函数访问到。

```js
//在某个组件中
store.commit({
    type:'increment',
    amount:10
})
//在store中的写法与上文一致
```

注意在mutation中我们不能使用异步函数，因为异步函数中发生的状态是不能追踪的，这样会让我们在调试时出现问题。我们通过action来与其他服务器交互，通过action使用异步函数。因此如果不需要异步操作的话，我们就可以直接越过action这个操作，但是需要异步是，我们需要先通过action，action会将状态的变化提交到mutation上，然后mutation进行处理，数据状态state中的内容发生改变，Vue组件将改变后的数据渲染进页面中。正如VueX声明周期描绘的那样。

#### 4）actions

action类似于mutation，在vue的组件中，我们可以通过action发起数据的状态改变，也可以通过mutation发起数据状态的改变，不同的是action最终会将数据提交给mutation，而不是直接变更数据的状态，但是我们在action中可以进行任意的异步操作。而mutation中不行。

##### 在store中注册action

每一个action的形式与mutation类似，都有一个事件名和回调函数，不同的是action中的函数传入的参数是context对象，我们通过`context.commit('your_mutation_name')`可以在action中提交一个mutation。也可以通过context来获取store的state和getter中的数据。但是context却不是store实例本身。

##### 在组件中使用action

action在组件中通过`store.dispatch('your_action_name')`触发，action和mutation一样，也有两种的传参方式，一种简单参数，一种是通过传入对象

### 3、模块化

如果全局的数据较多，都集中在一个store文件中，那么会非常难以管理，因此，VueX允许我们将store分割称模块，每个模块都拥有自己的state、mutation、action、getter，甚至还可以嵌套模块。我们可以对每个模块分别命名，然后在主模块中将这些子模块注入进去。那么我们访问每个模块中存放的数据的时候就需要在数据前面再加上模块名称

```js
const moduleA={
    state:()=>{},
    mutation:{},
    actions:{},
    getters:{}
}
const moduleB={
    state:()=>{},
    mutation:{},
    actions:{},
    getters:{}
}
const store = createStore({
    modules:{
		moduleA,
        moduleB
    }
})
//组件中访问
store.state.moduleA
store.state.moduleB
```

对于模块内部的mutation和getter接收的第一个参数是模块的局部状态对象，也就是说，该模块外部也就是说其他模块的state，是无法通过state访问到的。对于根状态通过第二或者第三个参数rootState访问。

同样对于模块内部的action，局部状态对象时通过context.state访问的，但是他仍提供了访问根状态的方法context.rootState

### 4、总结

如果要使用VueX，我们需要遵守一些规则：

* 应用层级的数据应该集中到store对象中
* 提交mutation是改变这些全局数据的唯一办法，而且提交内部中是同步的
* 异步的逻辑应该封装在action中
* 对于大型的应用，我们可以对store进行模块话改造，把action、mutation、getter等分割为单独的文件。

一些建议：

* 在开发环境中，我们可以开启严格模式，那么所有不是由mutation函数引起的全局数据的改变都会抛出错误，让我们更好的监测不合规的数据状态变更。但是我们不应该在生产模式使用它，因为性能损失较大。

## 十四、vue-loader

### 1、介绍







## 十五、element UI

### 1、自定义表单验证规则async-validator

#### 1）介绍

element-ui的表单组件中使用了自定义表单验证规则的第三方js库，如果我们需要对用户输入数据进行校验，我们需要了解这个库的具体用法。

#### 2）概念

在element-UI中我们需要在data中返回一个rules对象，这个对象内部装填有每个字段的验证规则，类似于async-validator官方说的定义一个descriptor。格式如下：

```json
rules: {
    password: [								   //假设对password字段进行有效性验证
        {type:string, required:true}			//验证规则对象
    ]
}
```

每个字段验证规则对象中允许的值有以下参数：

| 参数名         | 值                                                           | 说明               |
| -------------- | ------------------------------------------------------------ | ------------------ |
| type           | string\|number\|boolean\|method\|regexp\|integer\|float\|array\|object\|enum\|date\|url\|hex\|email |                    |
| required       | boolean类型                                                  | 是否为必填项       |
| pattern        | 正则表达式                                                   |                    |
| range(min,max) | min,max定义范围，字符串数组类型用于比较长度，数字类型比较大小 |                    |
| len            | integer类型，length优先级高于range                           | 验证字段确切长度   |
| enumerable     | 对于枚举类型，列出枚举的有效值                               |                    |
| fields         | 对于嵌套对象的属性值的验证，则需要在fields中为对象对应的属性定义规则 |                    |
| defaultField   | 值为对象或者数组，用来验证所有的值，是否满足defaultField规定的条件 |                    |
| transform      |                                                              |                    |
| messages       |                                                              |                    |
| asyncValidator | function(rule, value, callback)                              | 自定义异步验证函数 |
| validator      | function(rule, value, callback)                              | 自定义验证函数     |

## 附录1：Vue API

### 一、Vue实例

#### 1）Vue实例介绍

当一个vue实例对象被创建的时候，它将它内部的data对象的所有属性都加入到vue的响应式系统中，当这些属性的值发生改变时，视图就会产生响应。就更新为新的值。当这些数据改变的时候，视图就会重新进行渲染。只有当实例被创建是就存在在data中的属性才是响应式的。如果在实例创建之后在添加一个新的属性，那么这个属性的改动并不会触发视图的更新。所以我们需要在创建vue实例时，就定义好变量，只需要设置一些初始值。

用户定义的属性都会直接出现在vue的实例中，名字与用户定义相同。除了这些数据，vue实例还暴露了一些有用的属性和方法，是属于vue创造的，都有前缀$以便与用户定义的属性区分开。

#### 2）vue生命周期

每个vue实例在被创建时都需要经过一系列的初始化过程，直到实例被销毁。在这样的整个vue的生命周期中，有一些关键的时间点，会运行一些叫做生命周期钩子的函数。这就给了用户在不同阶段添加自己代码的机会。

![Vue 实例生命周期](https://cn.vuejs.org/images/lifecycle.png)

### 二、全局配置与全局API

#### 1）全局配置

针对Vue的全局配置指的是`Vue.config` 对象身上的属性和方法。

| 属性                  | 说明                                   |
| --------------------- | -------------------------------------- |
| silent                | 取消Vue所有日志和警告                  |
| optionMergeStrategies | ？？？                                 |
| devtools              | 是否允许开发者工具检查代码             |
| errorHandler          | 可以捕获和处理Vue指定组件的错误信息    |
| warnHandler           | 可以捕获和处理Vue指定组件的警告信息    |
| ignoreElements        | 忽略Vue之外的自定义元素                |
| keyCodes              | 给键盘事件自定义键位别名               |
| performance           | 可以在开发者工具中启用对组件的性能追踪 |
| productionTip         | Vue生产环境提示                        |

#### 2）全局API

| 方法 | 说明 |
| ---- | ---- |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |

### 三、Vue创建选项

指的是创建vue/vue组件对象时，你可以传入的参数。可以在创建对象的时候，指定该实例对象应该管理的东西如DOM对象、数据、子组件等，指定该实例对象自身的行为等。

#### 1）数据选项

| 属性/方法 | 说明                                                         |
| --------- | ------------------------------------------------------------ |
| data      | 这是vue实例绑定的数据对象。创建实例对象时，vue会递归的将data上的属性转换为getter和setter，从而实现响应式数据。实例创建之后，可以通过vm.$data.property访问原始的数据对象，也可以通过代理对象vm.property访问。以_或$开头的属性不会被vue实例代理，因此需要通过vm.$data来访问。 |
| props     | 用于接收来自父组件传递来的数据。可以使用简单的数组或者对象。也可以配置数据类型检测、设置默认值等。 |
| computed  | vue实例绑定的计算属性。通常需要依赖于data数据，只有data中对应数据更新后，它才会更新。 |
| methods   | 方法，在这里我们可以定义一些函数，然后通过vue实例对象使用。  |
| watch     | 响应数据的变化的另一种方式。                                 |

#### 2）DOM选项

vue绑定到HTML DOM对象有三种方法。一是通过CSS选择器、二是通过内部字符串模板、三是通过渲染函数生成HTML代码。这三种方法是互斥的。不能同时使用。

| 属性/方法 | 说明                                                         |
| --------- | ------------------------------------------------------------ |
| el        | 用于绑定vue外部HTML DOM 元素。只在创建实例时生效，对于实例对象需要使用$mount实例方法。 |
| template  | 在vue内部创建的HTML模板，该选项会**替换**已经绑定的HTML元素。如果 Vue 选项中包含渲染函数，该模板将被忽略。 |
| render    | HTML模板的代替方案，该渲染函数接收一个 `createElement` 方法作为第一个参数用来创建 `VNode`。 |

#### 3）生命周期钩子

生命周期及生命周期钩子的概念可以参见vue API 第一章——vue实例。

注意在所有生命周期钩子函数内部this自动绑定到vue实例对象（或者vue组件实例对象）中，我们就可以在生命周期函数内部访问实例对象的属性和方法。但是注意不要使用箭头函数定义生命周期函数。因为箭头函数的this绑定的是其上一级的实例对象，因此不一定会指向vue实例对象。

| 名称            | 解释                                                         |
| --------------- | ------------------------------------------------------------ |
| `beforeCreate`  | 在实例初始化之后，数据侦听、事件、侦听器配置之前             |
| `created`       | 在实例创建完成之后会立即调用。此时数据帧听、计算属性、方法事件的回调函数已配置完成 |
| `beforeMount`   | 注意到每个vue实例在其内部需要通过CSS选择器绑定到一个HTML模板上，然后才能对HTML DOM元素进行修改。这个动作官方称为挂载（mount），before mount意思就是vue实例绑定到对应的HTML模板上之前。 |
| `mounted`       | vue实例绑定到对应的HTML模板上之后，这时候对应的HTML元素可以通过vm.$el 访问。如果没有绑定到DOM元素，那么vue实例会处于未挂载状态 |
| `beforeUpdate`  | 在数据发生改变后，但是HTML DOM对象还没有被更新的时候会调用   |
| `updated`       | 在数据更新且HTML DOM对象已经被渲染更新完成后被调用           |
| `activated`     | 被 keep-alive 缓存的组件激活时调用，keep-alive指的是组件切换走的时候，不将这个组件销毁 |
| `deactivated`   | 被 keep-alive 缓存的组件失活时调用，主要用于关闭失活组件的某些方法，节省系统资源 |
| `beforeDestroy` | 在实例销毁之前会调用该函数，这个阶段我们可以做一些善后工作   |
| `destroyed`     | 实例销毁之后会被调用                                         |
| `errorCaptured` | 在子组件发生错误的时候会被调用。有三个参数：错误对象、发生错误的组件实例、错误信息 |

#### 4）资源与组合选项

| 名称           | 说明                                                  |
| -------------- | ----------------------------------------------------- |
| directives     | 用来定义Vue组件的多个自定义指令                       |
| filters        | 用来定义Vue组件的多个过滤器                           |
| components     | 用来定义Vue组件的多个子组件                           |
| parents        | 用来指定该Vue实例的父实例，建立两个组件之间的父子关系 |
| mixins         | 用来接收一个混入对象作为该vue实例的数据或属性或方法   |
| extends        | 用于创建一个子组件。                                  |
| provide/inject | ？？？？？                                            |

#### 5）其他选项

| 名称 | 说明          |
| ---- | ------------- |
| name | 给vue组件起名 |
|      |               |
|      |               |
|      |               |

### 五、实例属性与方法

实例属性和方法指的是在vue/vue组件对象创建完成后，生成的vue/vue组件对象，叫做实例对象。在实例对象身上，有一些属性和方法，叫做实例属性和实例方法。我们可以类比创建一个Java对象实例后，它身上具有的的属性和方法。如果要在JavaScript中访问或修改某个组件的数据，我们就可以通过其实例对象的属性和方法访问。

#### 1、实例属性

| 属性         | 详细                                                         | 其他       |
| ------------ | ------------------------------------------------------------ | ---------- |
| $data        | 当前组件所持有的数据                                         |            |
| $props       | 从父组件中接收到的数据                                       |            |
| $el          | 当前实例对象绑定的HTML DOM元素                               | 只读       |
| $options     | ？？？？                                                     | 只读       |
| $parent      | 当前组件的父组件对象，如果有的话。                           | 只读       |
| $root        | 当前组件树的根实例                                           | 只读       |
| $children    | 当前组件的直接子组件                                         | 只读       |
| $slots       | 用于访问被插槽分发的内容（在子组件中）                       | 只读非响应 |
| $scopedSlots | 用于访问扩展了作用域的插槽                                   | 只读       |
| $refs        | ？？？？                                                     | 只读       |
| $isServer    | 判断当前Vue实例是否是运行在服务器                            | 只读       |
| $attrs       | 包括了父组件中不作为prop绑定的其他数据(class style除外)，那么当一个组件没有声明过prop时，这里包含其父组件绑定的数据 | 只读       |
| $listeners   | 包含了父组件中v-on事件监听器                                 | 只读       |

#### 2、实例方法

##### 1）数据相关

| 方法    | 说明 |
| ------- | ---- |
| $watch  |      |
| $set    |      |
| $delete |      |

##### 2）事件相关

| 方法  | 说明                               |
| ----- | ---------------------------------- |
| $on   |                                    |
| $once |                                    |
| $off  |                                    |
| $emit | 发出信号，用于触发当前实例上的事件 |

##### 3）生命周期相关

在vue实例创建完成之后，我们也可以通过实例调用其生命周期关键点的一些方法，对实例做出进一步配置。

| 方法         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| $mount       | 如果vue在创建时没有绑定HTML DOM元素，那么它处于未挂载状态，我们可以通过该方法手动挂载一个元素。该方法返回实例本身。用于创建实例和挂载分离的情况。 |
| $forceUpdate | 迫使Vue实例重新渲染，它仅影响实例本身和插入插槽内容的子组件。 |
| $nextTick    | ？？？                                                       |
| $destory     | 用于销毁一个实例，清理它与其它实例的连接，解绑它的全部指令及事件监听器。 |

### 七、指令

| 指令      | 用法 |
| --------- | ---- |
| v-text    |      |
| v-html    |      |
| v-show    |      |
| v-if      |      |
| v-else    |      |
| v-else-if |      |
| v-for     |      |
| v-on      |      |
| v-bind    |      |
| v-model   |      |
| v-slot    |      |
| v-pre     |      |
| v-cloak   |      |
| v-once    |      |
|           |      |

#### 特殊指令属性





## 附录2：Vue Router API

### 一、Router HTML标签

#### 1、`<router-link>`



| 属性               | 作用 |
| ------------------ | ---- |
| v-slot             |      |
| to                 |      |
| replace            |      |
| append             |      |
| tag                |      |
| active-class       |      |
| exact              |      |
| event              |      |
| exact-active-class |      |
| aria-current-value |      |

#### 2、`<router-view>`



| 属性 | 作用 |
| ---- | ---- |
| name |      |

### 二、Router 创建选项

| 属性/方法                 | 说明 |
| ------------------------- | ---- |
| routes                    |      |
| mode                      |      |
| base                      |      |
| linkActiveClass           |      |
| linkExactActiveClass      |      |
| scrollBehavior            |      |
| parseQuery/stringifyQuery |      |
| fallback                  |      |

### 三、Router实例属性和方法

#### 1、实例属性



#### 2、实例方法



## 附录3：Vue X  API
