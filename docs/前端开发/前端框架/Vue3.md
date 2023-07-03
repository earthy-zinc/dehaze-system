# Vue3

## 简介

### API风格

#### 选项式API

使用选项式API，我们可以用包含多个选项的对象来描述组件的逻辑，选项所定义的属性都会暴露在函数内部的this上，他会指向当前组件实例。

#### 组合式API

通过组合式API，我们可以使用导入的API函数来描述组件逻辑。在单文件组件中，组合式API通常是会与`<script setup>`搭配使用，这个setup属性是一个标识，告诉vue在编译时进行一些处理，让我们可以更简洁的使用组合式API。这种API风格包含了以下方面的API。

* 响应式API：如ref和reactive，是我们可以直接创建响应式状态、计算属性和侦听器
* 生命周期钩子：让我们可以在组件各个生命周期阶段添加逻辑
* 依赖注入：如provide和inject，是我们在使用响应式API时，利用Vue的依赖注入系统

优点：

* 更好的逻辑复用
* 更灵活的代码组织
* 更好的类型推导
* 更小的生产包体积。`<script setup>`形式书写的组件模板被编译成了一个内联函数，和`<script setup>`中的代码位于统一作用于，就不需要像选项式API需要依赖this上下文对象访问属性，被编译的模板可以直接访问`<script setup>`中定义的变量，无需从实例中代理。这对代码的压缩更为友好。

## 创建Vue应用

每个Vue应用都是通过`createApp`函数创建一个新的应用实例，我们传入的createApp对象实际上是一个组件，每一个应用都需要一个根组件，其他组件将作为其子组件。应用实例必须在调用了`.mount()`方法后才会渲染出来，该方法接受一个容器参数，可以是一个实际的DOM元素或者是一个CSS选择器字符串。应用根组件的内容将会被渲染在容器根元素里面，容器元素自己不会被视为应用的一部分。`.mount()`方法应该始终在整个应用配置和资源注册完成之后被调用。他的返回值是根组件实例而不是应用实例。

应用实例会暴露一个`.config`对象允许我们配置一些应用级选项，例如顶用一个应用级错误处理器。用来捕获所有子组件上的错误。

```vue
app.config.errorHandler = (err) => {
  // 处理错误
}
```

应用实例还提供了一些方法来注册应用范围内可用的资源，比如注册一个组件。注册成功后，该组件在应用的任何地方都是可用的。

应用实例并不只限于一个，createApp允许我们在同一个页面中创建多个共存的Vue应用，而且每一个应用都拥有自己的用于配置和全局资源的作用域。

## 模板语法

在底层，Vue会将模板编译成高度优化的JavaScript代码，结合响应式系统，当应用状态变更时，Vue能够智能地推导出需要重新渲染的组件的最少数量，并应用最少的DOM操作。

### 文本内容插入

* 基本文本插入：最基本的数据绑定形式是文本插值，它使用的是Mustache双大括号语法。双大括号标签会被替换为相应组件实例中属性的值，同时，每次属性更改时他也会同步更新。
* 插入JavaScript表达式：Vue实际上在所有的数据绑定上都支持完整的JavaScript表达式。在Vue模板内，JavaScript表达式可以被用在文本插值中和任何Vue指令属性的值中。每个绑定仅支持单一的表达式，可以在绑定的表达式中使用一个组件暴露的方法，也就是调用函数。模板中的表达式被沙盒化，仅仅能够访问到有限的全局对象列表，该列表会暴露常用的内置全局对象。没有显示包含在列表中的全局对象将不能在模板表达式中访问。

### HTML文本插入

双大括号会将数据解释为文本，而不是HTML，如果想要插入HTML，我们需要使用v-html指令，这里指令由`v-`作为前缀，表明他们是一些由Vue提供的特殊HTML标签属性，这些指令为渲染的DOM应用特殊的响应式行为。

### 绑定HTML属性值

* 常规HTML属性的绑定：双大括号不能再HTML属性中使用，如果想要响应式的绑定一个属性，应该使用v-bind指令。v-bind如果绑定的是null或者undefined，那么对应HTML标签上的该属性会从渲染的元素中移除。由于这个指令非常常用，因此有简写写法`:`开头为`:`的HTML属性是合法的属性名称。

* 对于布尔型的HTML属性，根据其值为true或者false来决定属性是否应该存在于该元素上，disabled属性就是最常见的例子。
* 一次绑定多个HTML属性：通过不带参数的v-bind我们可以将多个属性使用一个`v-bind="objectAttrs"`绑定到单个HTML标签上。

### 指令

指令是带有`v-`前缀的特殊属性，Vue提供了许多内置指令，指令属性的期望值是一个JavaScript表达式，一个指令的任务是在其表达式的值变化时响应式的更新DOM。

#### 指令参数

某些指令会需要一个参数，在指令名称后面通过一个冒号隔开做标识。这个参数通常是HTML原生的属性或者事件名

#### 动态指令参数

指令我们也可以使用动态传入的方式指定。如

```vue
<a v-bind:[attrName]="url">...</a>
```

这里的attrName会作为一个JavaScript表达式被动态执行，计算得到的值会被用作最终的参数。相似的，我们还可以将一个函数绑定到动态的事件名称上。动态参数中的表达式的值应当是一个字符串或者null，null表示移除该绑定。字符串中不应该有空格或引号。

#### 指令修饰符

修饰符是以点`.`开头的特殊后缀，表明指令需要以一些特殊的方式被绑定，例如`.prevent`修饰符会告知v-on指令对出发的事件调用某一函数。

## 响应式编程

### 声明响应式状态

使用reactive()函数创建一个响应式对象或者数组，响应式对象实际上是JavaScript Proxy，其行为表现与一般对象类似，但是不同之处在于Vue能够跟踪对响应式对象属性的访问和更改操作。

reactive()函数能够隐式的从参数中推导类型，但是我们可以使用接口显式标注一个reactive响应式变量的类型。

```typescript
interface Book {
  title: string,
  year?: number
}
const book: Book = reactive({title: 'vue'})
```

当我们更改响应式状态后，DOM文档对象模型会自动更新。但是DOM的更新不是同步的，Vue会缓冲修改的状态直到更新周期的下一个时机。如果要等待一个状态改变后的DOM更新完成后进行一些操作，需要使用nextTick()函数。

在Vue中，状态默认为深层响应式，意味着即使在更改深层次的对象或数组，改动也能够检测到。但是reactive()返回的是原始对象的代理，而不是原始对象本身，也就是说，只有代理对象是响应式的，更改原始对象不会触发更新。对同一个原始对象调用reactive()总是会返回相同的代理对象，而对一个已存在的代理对象调用则会返回其本身。依靠深层响应性，响应式对象内的嵌套对象依然是代理。

但是reactive()仅仅对对象类型是有效的，而对原始类型如string、number、boolean无效。因为Vue响应式是通过属性访问来追踪的，因此必须始终保持对该响应式对象的相同引用。这意味着当我们将响应式对象属性赋值到本地变量、或者通过结构赋值的方式赋值到本地变量，或者将该属性传给一个函数中，会失去响应性。

reactive()的限制是因为JavaScript没有可以作用于所有值类型的引用机制，因此，Vue提供了一个ref()方法允许我们创建可以使用任何值类型的响应式代理ref。ref()会将传入参数的值包装为一个带.value属性的ref对象。一个包含对象类型值的ref可以响应式地替换整个对象。ref被传递给函数或者从一般对象上被结构的时候，不会丢失响应性。

当ref响应式对象作为顶层属性，也就是再模板中使用时，他们会自动解包，不需要再带有.value。

当ref被嵌套在一个响应式对象中，作为属性被访问或者被更改时，他会自动捷豹，因此会表现的和一般属性一样/

## 计算属性

定义计算属性，用于对对象进行计算处理。

```javascript
const bookMessage = computed(() => {
	return author.book.length > 0 ? "Yes":"No";
})
```

定义计算属性，只需要采用computed()方法，方法接受一个getter函数，返回值为一个计算属性响应式对象ref。我们可以通过.value访问计算结果，计算属性在模板中也会自动解包，无需添加.value。计算属性可以自动追踪响应式依赖，当他检测到原始所依赖的对象发生改变时，他自身也会变化。

当然我们通过方法也可以返回一个和计算属性同样的结果，不同的是，计算属性会基于其响应式的依赖被缓存，一个计算属性只有在其依赖的原始对象被更改时才会重新计算，也就是说只要原始对象不改变，无论访问多少次计算属性，都会立即返回先前计算的结果。相比之下方法调用总是会在重新渲染时再次执行函数。

计算属性默认是只读的，只有在特殊场景中，才会需要可写这种属性，我们可以给computed方法提供一个对象，对象中带有get和set方法来创建可写的计算属性

## 类与样式绑定

数据绑定的常见需求是操纵元素的CSS类列表和内联样式，因为class类和style样式都是HTML元素的属性，我们可以向其他属性一样使用v-bind将他们和动态的字符串绑定，但是在处理较为复杂的绑定时，通过拼接生成字符串是麻烦还容易出错的，因此，针对class和style这两种属性，vue提供了特殊增强，除了字符串以外，表达式的值也可以是对象或者数组。

### 绑定HTML元素的class属性
#### 给class属性绑定一个对象

```vue
<template>
<div :class="{active: isActive}"></div>
</template>
```

上述语法表示html中该div元素的class属性中的active子属性存在与否取决于布尔值变量isActive的的真假。

注意`:class(v-bind:class)`指令可以和普通的`class`属性共存。在虚拟DoM渲染后，会将两个属性合并为一个。当class属性中的子属性变化时，class属性列表也会随之更新。

当然绑定的对象不一定需要写成内联的字面量形式，也可以直接给class属性绑定一个响应式对象，响应式对象的格式应和上述的相同，其中对象的属性名应该是css类class的名字，值是一个布尔值，表示该css类是否应该加到当前html元素class属性上。当然绑定的响应式对象可以是计算属性。

#### 绑定数组

我们也可以给class绑定一个数组来渲染多个css的class

```vue
<template>
<div :class="[activeClass, errorClass]"></div>
</template>
<script>
const activeClass = ref("active");
const errorClass = ref("text-danger");
</script>
```

渲染的结果是：

```html
<div class="active text-danger"></div>
```

如果想要在数组中有条件的渲染某个CSS class，我们可以使用三元表达式。

### 组件上绑定class

如果有一个父组件，想要给其子组件传递一些HTML元素上的属性，则需要属性继承

对于只有一个根元素的组件，当我们使用了class属性是，这些属性会被添加到根元素上，并且与该根元素已经存在的class属性合并。

如果组件中存在多个根元素，你需要指定哪一个元素来接受来自父组件传来的class属性，我们可以通过组件上内置的`$attrs`属性来实现指定

子组件MyComponent：

```vue
<template>
  <p :class="$attrs.class">Hi!</p>
  <span>This is a child component</span>
</template>
<script>
  export default {
    name: MyComponent
  }
</script>
```

父组件

```vue
<template>
<MyComponent class="baz"/>
</template>
```

该子组件将被渲染为：

```html
<p class="baz">Hi!</p>
<span>This is a child component</span>
```

### 绑定内联样式






## 条件渲染

## 列表渲染

## 事件处理

## 表单输入绑定

## 生命周期

 每个Vue组件实例在创建时都需要经历一系列初始化步骤，比如设置好数据侦听，编译模板，挂载实例到DOM，以及在数据改变时更新DOM文档，在这个过程中，他也会运行被称为生命周期钩子的函数，让开发者有机会能够在特定阶段运行自己的代码。

### 注册生命周期钩子

比如，onMounted钩子可以用来在组件完成初始渲染，并且创建DOM结点之后运行代码。当调用onMounted时，Vue会自动将回调函数注册到当前正在被初始化的组件实例上，这意味着这些钩子应该会在组件初始化时被同步注册。

## 侦听器

计算属性允许我们声明性地计算一些衍生的值，然是在有些情况下，我们需要在某些状态发生变化时，执行一些事情，如更改DOM，或者根据异步操作的结果修改另一处的状态。

在组合式API中，我们可以使用watch函数在每次响应式状态发生变化时，触发一个回调函数，执行一些事情。

```vue
<script>
	const question = ref("");
  const	answer = ref("question");
  watch(question, async (newQuestion, oldQuestion)=>{
    if(newQuestion.indexOf('?')>-1){
      answer.value = 'Thinking';
    }
  })
</script>
```

### 侦听器侦听数据源的类型

watch侦听器是一个方法，用于侦听他的第一个参数发生的变化，然后调用给定第二个参数传入的方法执行做一些事情。

第一个参数可以是不同类型的数据源，如ref、计算属性、响应式对象、getter函数，多个数据源组成的数组。

但是不能直接侦听响应式对象的属性值，而是需要用一个返回该属性的getter函数

```vue
<script>
	const obj = reactive({ count: 0 });
  watch(
    () => obj.count,
    (count) => {
      console.log(`count is: ${count}`)
    }
)
</script>
```

直接给侦听器watch()传入一个响应式对象，会隐式地创建一个深层侦听器，该回调函数在所有嵌套值变更时都会被出发。相比之下，一个返回响应式对象的getter函数，只有getter函数的返回值返回不同的对象时，才会触发回调。

watch侦听器默认是懒执行的，仅当数据源变化时，才会执行回调。但是在某些场景中，我们希望在创建侦听器后，立即执行一遍回调，

## 模板引用

虽然Vue声明性渲染模型抽象了大部分对DOM的直接操作，但是在某些情况下，仍然需要直接访问底层DOM元素，我们可以使用ref属性，它允许我们在一个特定的DOM元素或者子组件实例被挂载后，获得对他的直接引用，比如说能够在组件挂载时将焦点设置到一个input元素上，或者在一个元素上初始化一个第三方库。

### 通过组合式API访问对模板对象的引用

为了获得在HTML模板DOM文件中对应元素的引用，我们需要在script脚本中声明一个同名的ref。只有在组件挂载之后才能够访问模板应用，在Vue对HTML页面的初次渲染之前该元素还不存在。

当在v-for中使用模板的引用时，对应的ref值为一个数组。

```vue
<script setup>
	import { ref, onMounted } from 'vue'
  
  const itemRefs = ref([])
  onMounted(() => console.log(itemRefs.value))
</script>
<template>
  <ul>
    <li v-for="item in list" ref="itemRefs">
      {{ item }}
    </li>
  </ul>
</template>
```

### 使用函数值作为引用元素的名字

ref绑定的元素对象，除了 使用字符串作为这个DOM对象的名字，ref属性还可以通过v-bind绑定为一个函数，会在每次组件更新时都被调用。该函数会收到元素引用作为其第一个参数。也就是说绑定的那个函数的第一个参数就是该dom元素对象。当绑定的dom元素被卸载时，函数也会被调用一次，这是传入的参数就为null。

### 绑定组件

模板引用也可以被用在一个子组件上，这种情况下引用中获得的值不再是DOM对象而是组件实例。

## 组件

### 组件基础

组件允许我们将UI划分为独立的、可重用的部分，并且可以对每一部分进行单独的思考，Vue实现了自己的组件模型，使我们可以在每个组件内封装自定义内容和逻辑，Vue也能够配合原生的网页组件。

每当使用一个组件，就创建了一个新的实例，在单位见组件中，我们为子组件使用首字母大写的标签名，以此和原生HTML元素做区分，虽然原生HTML标签名是不区分大小写的，但是Vue单文件组件是可以在编译中区分大小写的。

#### 动态组件

如果想要在多个组件之间来回切换，需要用到动态组件。这里是通过Vue的`<component>`元素和特殊的is属性实现的

 ```vue
 <template>
   <!-- currentTab 变量值改变时组件也改变 -->
   <component :is="tabs[currentTab]"></component>
 </template>
 ```

可以传递给:is的有以下几种：

* 被注册的组件名
* 导入的组件对象



### 组件注册

一个组件在使用之前需要先注册，然后再渲染模板时才能找到其对应的实现，组件注册有两种方式：全局注册和局部注册

#### 全局注册

使用Vue应用实例的app.component()方法，让组件再当前Vue全局可用

```js
import { createApp } from 'vue'
import MyComponent from './App.vue'
const app = createApp({})

app.component('MyComponent', MyComponent)
```

全局注册的组件可以在此应用的任意组件模板中使用。

#### 局部注册

全局注册但是并没有被使用的组件无法再生产打包时被自动移除，如果注册了一个全局组件，即使没有用，仍然会出现在打包后的JS文件中。

在使用 `<script setup>` 的单文件组件中，导入的组件可以直接在模板中使用，无需注册

```vue
<script setup>
import ComponentA from './ComponentA.vue'
</script>

<template>
  <ComponentA />
</template>
```

#### 组件名称规范

组件名称需要使用首字母大写的格式，这是合法的JavaScript标识符，使得在JavaScript中导入和注册组件都变得很容易，这种形式的名称可以和原生的HTML元素区分开，将Vue组件和自定义元素区分开。

### props

一个组件需要显式的声明它所接受的props，props通常是父组件传递给子组件的一些数据，如果不显式的声明，Vue就不太明白外部组件传入的究竟是props还是透传attribute

#### 向组件中传递对象数据

props是一种特殊的虚拟HTML元素属性，我们可以在子组件内部生命注册一个props，使用defineProps进行声明。defineProps是仅仅在`<script setup>`中可用的命令，不需要显式导入，声明的props会自动暴露给模板，他会返回一个对象，包含了父组件中传递给该子组件的所有信息。

子组件*BlogPost.vue*

```vue
<script setup>
const props = defineProps(['title'])
</script>

<template>
  <h4>{{ props.title }}</h4>
</template>
```

父组件：

```vue
<template>
  <BlogPost title="My journey with Vue" />
  <BlogPost title="Blogging with Vue" />
  <BlogPost title="Why Vue is so fun" />
</template>
```





### 事件

子组件如果需要和父组件进行交互，比如一个控制按钮在某个子组件中，这个按钮可以实现让父组件中所有的字体都放大。那么子组件在自身应该定义一个事件，并且在用户点击该按钮时，将事件传递给父组件。组件实例提供了一个自定义事件系统，父组件可以通过v-on 或@来选择性的监听子组件上抛的事件，就像监听原生DOM事件那样。

#### 子组件抛出事件

子组件通过调用内置的`$emit`方法，传入事件名称(字符串)抛出一个事件（$emit方法无法在`<script setup>` 中调用，可以使用defineEmits代替）

```vue
<!-- BlogPost.vue, 省略了 <script> -->
<template>
  <button @click="$emit('enlarge-text')">
    Enlarge text
  </button>
</template>
```

子组件也可以通过defineEmits宏来声明子组件需要抛出的事件，声明了需要抛出的事件后，就可以对事件的参数进行验证，同时还可以让避免将他们作为原生事件监听器应用于子组件根元素。defineEmits会返回一个等同于$emit方法的emit函数，我们可以调用它来触发事件。

#### 父组件监听事件

父组件的虚拟HTML元素的属性上，需要填写子组件中定义的事件名称（字符串）来监听子组件抛出的对应事件信息。这个事件名称是子组件中通过`$emit()`定义的事件名。父组件对应的属性值则填写JavaScript表达式，触发的事件会执行这段代码。父组件通过v-on缩写为@来监听事件，组件事件监听器也支持修饰符。

```vue
<template>
<BlogPost 
  @enlarge-text="postFontSize += 0.1"/>
</template>
```

#### 事件参数

有时我们需要在触发事件时附带一个特定的值，宅这种场景下，我们可以给子组件`$emit` 提供一个额外的参数。然后我们在父组件中监听事件，父组件定义的函数就能够收到子组件`$emit`方法提供的参数。所有传入`$emit()`的额外参数都会被直接传向监听器。

#### 声明触发的事件

我们在模板HTML中使用的`$emit`方法不能再组件的`<script setup>`中使用，但是defineEmits()能够返回一个相同作用的函数使用，但是defineEmits不能再子函数中使用，必须直接放置在`<script setup>`的顶级作用域下。

emit选项支持对象语法，允许我们对触发事件的参数进行验证。

```vue
<script setup>
	const emit = defineEmits({
    submit(payload){
      // 通过验证返回值为true false来判断是否通过
    }
  })
</script>
```

#### 事件校验

所有触发的事件也可以用对象的形式进行描述。要为事件添加校验，那么事件可以被赋值为一个函数，接收的参数就是抛出事件时传入emit的内容，返回一个布尔值来表明事件是否合法。

 ```vue
 <script setup>
 	const emit = defineEmits({
     // null 表示不进行参数校验
     click: null,
     // 校验submit事件传入的email和password参数
     submit: ({email, password}) => {
       // && 短路操作，表示这两个参数都不为null或undefined
       if (email && password){
         return true;
       }else {
         console.warn("不能为空");
         return false;
       }
     }
   })
   
   function submitForm(email, password){
     emit('submit', {email, password})
   }
 </script>
 ```

### 组件 v-model

v-model可以在组件上使用以实现双向绑定。首先v-model在原生组件中，模板编译器会对v-model进行等价展开。实际上等价于

```html
<input v-model='searchText'/>
<input :value='searchText'
       @input="searchText = $event.target.value"
```

也就是说，v-model绑定了一个value单项绑定，又绑定了一个input事件将事件中的该对象的value值绑定到了searchText变量中。而在应用到一个组件中，v-model会被展开为以下形式：

```html
<CustonInput :modelValue="searchText"
             @update:modelValue="newValue => searchText = newValue"/>
```

这里这个组件将内部原生的input元素的value属性绑定到了modelValue的prop值中，当原生的input事件触发后，触发了一个携带了新值的事件

#### v-model参数

默认情况下，v-model在组件上都是使用modelValue作为prop，并且以modelValue作为对应的事件。我们可以通过给v-model指定一个参数来更改这些名字。利用这种特性，我们可以在单个组件实例上创造多个v-model双向绑定。

父组件中：

```vue
<UserName v-model:first-name='first'
          v-model:last-name='last'/>
```

子组件中：

```vue
<script setup>
	defineProps({
    firstName: String,
    lastName: String
  })
  
  defineEmits(['update:firstName, update:lastName'])
</script>
<template>
	<input type='text'
         :value='firstName'
         @input="$emit('update:first', $event.target.value)"/>

</template>
```

####  处理修饰符

v-model有一些内置的修饰符，例如.trim .number .lazy在某些场景下，可能想要自定义组件的v-model支持自定义的修饰符。组件的v-model上所添加的修饰符，可以通过modelModifiers 这个prop在组件内访问到。在下面的组件中，声明了这个prop，默认值是一个空对象。有了这一个prop，我们就可以检查modelModifiers对象上的键，并编写一个处理函数来改变抛出的值。

```vue
<script setup>
	const props = defineProps({
    modelValue: String, 
    modelModifiers: { default: ()=> ({})}
  })
  const emit = defineEmits(['update:modelValue'])
  
  function emitValue(e){
    let value = e.target.value
    if(props.modelModifiers.captialize){
      value = value.charAt(0).toUpperCase() + value.slice(1)
    }
  }
</script>
```




### 透传Attributes

透传属性指的是传递给一个组件，却没有被该组件声明为props或者emits的属性或者v-on事件监听器，常见的例子是class、style、id。简单来说，透传意味着父组件或者外部组件传递给当前组件的属性，这些是一般而言的属性，而没有没vue做其他特殊处理。

当一个组件以单个元素为根做渲染时，透传的属性会被自动添加到该组件的根元素上。

#### 对于class、style属性

对于父组件或者外部组件传递给当前组件的style或者class属性。这些属性会和当前组件已有的style和class属性合并。

#### 对于v-on绑定的监听器

如果父组件给子组件绑定了一个监听器属性，那么这个监听器就会被添加到子组件的根元素，当子组件中当前事件被触发，就会触发父组件中的对应事件方法。

#### 深层继承属性

如果子组件只是在根元素结点上渲染另一个组件，而没有其他操作，那么父组件传递来的属性就会直接再次传递给子子组件。

如果不需要组件自动的从父组件继承属性，可以在组件选项中设置`inheritAttrs: false`，我们在 `<script setup> `中使用 defineOptions来设置这个值。

最常见的需要禁用属性继承的场景是，属性需要应用在根节点意外的其他元素上，通过这样设置，我们可以控制传递进来的属性应该被如何使用，这些传递进来的属性可以在模板表达式中直接用`$attrs`访问到。这里面包含了除了组件所声明的props和emits之外的所有其他属性。

比如我们想把属性应用到内部button结点上，这样使用`v-bind="$attrs"`实现。没有参数的v-bind会将对象的所有属性都作为html元素属性应用到html元素结点上。

#### 多个根节点的属性继承

和单根节点的组件有所不同，有多根节点的组件没有自动的继承传递属性的行为，如果$attrs没有被某个元素显式的绑定将会抛出一个运行时警告。

#### JavaScript中获取继承传递的属性

在 `<script setup> `中使用 `useAttrs()` API 来访问一个组件的所有透传 attribute：

### 插槽

向子组件传递大量内容

之前我们介绍了可以通过子组件元素属性向子组件传递一些对象数据，有时我们希望传递的信息量大，或者向子组件传递HTML元素，等大量的信息，那通过HTML属性传递就比较麻烦，这是我们可以通过插槽来传递大量内容。

#### 父组件传递内容

父组件在子组件元素的text域内填写想要传递的内容

```vue
<template>
  <AlertBox>
    Something bad happened.
  </AlertBox>
</template>
```

#### 子组件接受内容

子组件可以在HTML模板的某个位置添加自定义的`<slot>`元素来接收父组件传递的内容

```vue
<template>
  <div>
    <strong>This is an Error for Demo Purposes</strong>
    <slot />
  </div>
</template>
```

`<slot>`元素是一个插槽出口，表示了父元素提供的插槽内容将会在哪里被渲染。通过使用插槽，子组件仅负责渲染外层的html组件以及相应的样式，而内部的内容则有父组件提供。插槽的内容可以是任何合法的模板内容，不局限于文本，比如我们可以传入多个元素，甚至是组件。通过使用插槽，子组件就更加灵活和具有可复用性，现在组件也可以用在不同的地方用来渲染各异的内容，同时还保证都具有相同的样式。

#### 渲染的作用域

插槽内容可以访问到父组件的数据作用域，因为插槽内容本身就是在父组件模板中定义的，插槽内容无法访问子组件的数据，Vue模板中的表达式只能访问其定义时所处的作用域，这和JavaScript的词法作用域规则是一致的

#### 默认内容

在外部没有提供任何内容的情况下，可以为插槽指定一个默认内容。也就是说父组件未向子组件传递内容，那么子组件`<slot>`标签之间的内容就会被当作默认内容渲染。

#### 有名插槽

有时候一个子组件中需要有多个插槽出口，如果我们想要分别不同的插槽，就需要名字来对他们进行区分。对于这种场景，`<slot>`元素有一个特殊属性name，用于给每个插槽分配一个唯一的ID，这类带有name的插槽被称为有名插槽，没有提供name属性的插槽会被默认命名为default。在父组件中使用带有插槽的子组件时，我们就需要一种方式将多个插槽内容传入到各自目标插槽的出口，此时需要用到具名插槽。

要为有名插槽传入内容，我们需要一个含有`v-slot`指令的`<template>`元素，并且将目标插槽的名字传给该指令。`v-slot`有对应的简写为`#`。因此`<template v-slot:header>`可以简写为`<template #header>`，其意思就是将这部分模板片段传入子组件的header插槽中。

当一个组件同时接收默认插槽和具名插槽时，所有位于顶级的非`<template>`节点都会被隐式的是为默认插槽的内容。

因此这两段代码是等同的。

```vue
<template>
    <BaseLayout>
      <template #header>
        <h1>Here might be a page title</h1>
      </template>

      <template #default>
        <p>A paragraph for the main content.</p>
        <p>And another one.</p>
      </template>

      <template #footer>
        <p>Here's some contact info</p>
      </template>
    </BaseLayout>
</template>
```

```vue
<template>
    <BaseLayout>
      <template #header>
        <h1>Here might be a page title</h1>
      </template>

      <!-- 隐式的默认插槽 -->
      <p>A paragraph for the main content.</p>
      <p>And another one.</p>

      <template #footer>
        <p>Here's some contact info</p>
      </template>
    </BaseLayout>
</template>
```

#### 动态插槽名称

动态指令参数在`v-slot`上也是有效的，也就是说可以定义下面这样动态插槽名。

```vue
<template>
	<BaseLayout>
    	<template v-slot:[dynamicSlotName]>		
		</template>

        <template #[dynamicSlotName]>		
		</template>
    </BaseLayout>
</template>
```

#### 作用域插槽

插槽作为父组件传递给子组件的内容，插槽是无法访问到子组件的状态的，然而在某些场景中，插槽的内容可能想要同时使用父组件领域内和子组件领域内的数据，要做到这一点，我们需要让子组件在渲染时，将一部分数据提供给插槽。

我们可以向对组件传递属性、props、事件那样，向一个插槽出口上绑定属性。

在子组件中，我们绑定了text、count属性，text属性的值由变量greetingMessage确定。

```vue
<template>
	<div>
        <slot :text="greetingMessage"
              :count="1"></slot>
    </div>
</template>
```

在父组件中，当需要接受子组件从插槽内传递出来的数据时，我们先看看默认插槽如何接受数据。首先通过子组件标签上的`v-slot`指令，直接接收到了一个插槽props对象。

```vue
<template>
	<MyComponent v-slot="slotProps">
    	{{ slotProps.text }}
        {{ slotProps.count }}
    </MyComponent>
</template>
```

### 依赖注入

通常情况下，当我们需要从父组件向子组件传递数据时，会使用props。如果根组件需要给某个子子子子组件传递数据，如果仅仅使用props则必须沿着组件链逐级传递下去，这回非常麻烦，并且中间的组件并不关心这些数据，但是为了让数据传递，他们仍然需要定义这些props并向下传递，我们需要避免这种情况。

provide和inject可以帮助我们解决这一问题，一个父组件相对于其所有的后代组件，会作为依赖提供者，任何后代组件树，无论层级有多深，都可以注入由父组件提供给整条链路的依赖。

#### provide

要为组件后代提供数据，需要用到provide函数。provide函数接收两个参数，第一个参数被称为注入名，可以是一个字符串或者是一个symbol。后代组件会用注入名来查找期望注入的值。一个组件可以多次调用provide()使用不同的注入名，注入不同的依赖值。第二个参数是提供的值，可以是任意类型，包括响应式状态。

 ```vue
 <script setup>
 	import {provide} from 'vue'
   provide('message', 'hello!')
 
 </script>
 ```

除了在一个组件中提供依赖，我们还可以在整个应用层面提供依赖，在应用级别提供的数据在该应用内的所有组件中都可以注入。

#### inject

要注入上层组件提供的数据，需要使用inject()函数。如果提供的值是一个响应式对象，注入进来的会是该响应式对象，不会自动解包为其内部的值，这使得注入方组件能够通过ref对象保持和供给方的响应式链接。

```vue
<script setup>
	import { inject } from 'vue'
  const message = inject('message')
</script>
```

默认情况下，inject假设传入的注入名会被某个祖先链上的组件提供，如果该注入名的数据没有任何组件提供，则会抛出一个警告，如果在注入一个值的时候不要求必须有提供者，那么我们应该声明一个默认值，在一些情况下，默认值可能需要通过调用一个函数或者初始化一个类来却，为了避免在用不到默认值的情况下进行不必要的计算，我们可以使用工厂函数来创建默认值。

当提供或者注入响应式数据时，我们尽可能将任何对响应式状态的变更都保持在供给方组件中，这样可以确保所提供的状态声明和变更操作都内聚在一个组件中，使其更加容易维护。

但有时候，我们可能需要在注入方组件中更改数据，在这种情况下，推荐在供给方组件内声明并提供一个更改数据的方法函数。

供给方组件

```vue
<script setup>
	import {provide, ref} from 'vue'
  const location = ref("pole");
  function updateLocation(){
    location.value = "south pole"
  }
  
  // 提供一个响应式数据location，并提供他的更新方法
  provide('location', {
    location, 
    updateLocation
  })
</script>
```

接收方（注入方）

```vue
<script setup>
	import {inject} from 'vue'
  const {location, updateLocation} = inject('location')
</script>
<template>
	<button @click='updateLocation'>
    {{ location }}
  </button>
</template>
```

如果想要确保提供的数据不被接收方更改，可以使用readonly来包装提供的值

#### 使用symbol作为注入的名

如果包含非常多的依赖注入，普通的字符串当注入名就不够用了，这是可以使用Symbol来作为注入以避免潜在的注入冲突。在单独的文件中导出这些注入名Symbol

```js
export const injectionKey = Symbol()
```

供给方组件

```js
import { provide } from 'vue'
import { injectionKey} from './key.js'
provide(injectionKey, {
  
})
```

注入方

```js
import { inject } from 'vue'
import { injectionKey } from './key.js'

const injected = inject(injectionKey)
```





### 异步组件

在大型项目中，我们可能需要拆分应用为更小的块，并且仅在需要的时候再从服务器加载相关组件。Vue提供了`defineAsyncComponent`方法来实现此功能。

```js
import { defineAsyncComponent } from 'vue'
const AsyncComponent = defineAsyncComponent(() => {
  return new Promise((resolve, reject) => {
    // 从服务器获取组件
    resolve(/*返回处理获取的组件*/)
  })
})
```

ES模块动态导入也会返回一个Promise，多数情况下将它和defineAsyncComponent搭配使用，最后得到的值就是一个外层包装过的组件，仅仅再页面需要它渲染时才会调用加载内部实际组件的函数，他会将接收到的props和插槽传给内部组件。

与普通组件一样，异步组件可以使用app.component()全局注册，也可以直接再父组件中定义。

异步操作不可避免会涉及到加载和错误状态，因此defineAsyncComponent()也可以处理这些状态。

### 内置组件

## 组合式函数

组合式函数是利用Vue的组合式API来封装和复用有状态逻辑的函数，当构建前端应用时，我们常常需要复用公共任务的逻辑，如为了在不同地方格式化时间，我们可能会抽取一个可以复用的日期格式化函数，这个函数封装了无状态逻辑，他在接收输入后会立刻返回所期望的输出。相比之下有状态逻辑会管理随着时间变化的状态。

## 自定义指令



## 插件



## 应用规模化



## 服务端渲染

### 总览

默认情况下，Vue组件的职责是在浏览器生成和操作DOM，然而，Vue也支持将组件在服务端直接渲染成HTML字符串，作为服务端响应数据，返回给浏览器，最后在浏览器端，将静态的HTML激活（hydrate）为能够交互的客户端应用。

Server-Side Rendering（服务端渲染）的优势：

* 更快的首屏加载：服务端渲染的HTML无需等到所有的JavaScript都下载并执行完成之后才显示，所以用户将会更快的看到完整渲染的页面。数据获取过程在首次访问时在服务端完成，相比于从客户端获取，会有更快的数据库连接。
* 更快的搜索引擎优化（Search Engine Optimization, SEO）