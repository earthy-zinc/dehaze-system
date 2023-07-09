# Pinia 学习

## 定义store

store是vue数据的仓库，在setup中是通过`defineStore()`定义的，他的第一个参数要求是一个独一无二的名字。这个名字也被用作id，是必须传入的，我们将defineStore()返回的一个函数命名为`use...`是一个符合组合式函数风格的约定。第二个参数可接受两个值，setup函数或者option对象。

option对象，可以带有state、actions、getters属性。state是数据仓库中的数据，而getters是数据仓库中的计算属性，actions则是方法。

```js
const useCounterStore = defineStore('counter', {
  state: () => ({count:0}),
  getter: { 
  	double: (state) => state.count * 2.
  },
  actions: {
    increment() {
      this.count++
    }
  }
})
```



当然也存在另一种定义store的方法，我们可以在第二个参数传入一个函数，该函数定义了一些响应式属性和方法，并且返回一个带有我们想暴露出去的属性和方法的对象。

```js
const useCounterStore = defineStore('counter', ()=>{
  const count = ref(0);
  function increment() {
    count.value++
  }
  return {count, increment}
})
```

在setup store中。

* ref()相当于state属性
* computed()相当于getter属性
* function()相当于actions属性

像这样的setup数据仓库比option数据仓库带来了更多的灵活性，可以在数据仓库中创建侦听器，并且自由地使用任何组合式函数，但是会让服务端渲染更加复杂。

## 使用store

我们要在使用数据仓库的组件中调用该use...Store()来获取数据仓库变量，在调用之前，store实例是不会被创建的。我们可以定义任意多的store，但是最好在不同的文件中定义store。一旦store实例被创建，我们就可以直接访问在store中定义的state、getters、actions等属性。为了从store中提取属性时保持其响应性，我们应该使用storeToRefs()。它将为每一个响应式属性创建引用，当我们只是用状态而不调用action函数时，会很有用。

```vue
<script setup>
	import { storeToRefs } from 'pinia'
  const store = useCounterStore();
  const {name, doubleCount} = storeToRefs(store);
  const {increment} = store;
</script>
```

## State

state即数据，state被定义为返回初始状态的函数。

```typescript
const useStore = defineStore('storeId', {
  state: () => {
    return {
      count: 0,
      name: 'jack',
      isAdmin: true,
      items: [] as UserInfo[],
      user: null as UserInfo | null
    }
  }
})

interface UserInfo {
  name: string, 
  age: number
}
```

 当然我们也可以用接口来定义state的返回值，并且定义state函数返回值的类型。

在默认情况下，我们可以通过store实例来访问state数据，直接对其进行读写。也可以调用store的$reset() 方法将state重置为初始值。

### 变更state

除了使用store.count++直接改变store，我们还可以调用$patch方法，使用一个state的补丁对象再同一时间修改多个属性。

```js
store.$patch({
  count: store.count + 1,
  age: 120,
  name: 'DIO',
})
```



### 侦听state

我们可以通过store的`$subscribe()`方法侦听state及其变化，使用这个方法的好处时订阅方法在patch分发后只触发一次。

```js
cartStore.$subscribe((mutation, state) => {
  mutation.type
  mutation.storeId
  mutation.payload
  localStorage.setItem('cart', JSON.stringify(state))
})
```

默认情况下，state subscription会被绑定再添加他们的组件上，这意味着当该组件被卸载时，对这些数据变化的订阅消息会被自动删除，如果想要再组件卸载后依然保留，需要将detached: true作为第二个参数

## Getter

getter完全等同于store的state计算值，可以通过defineStore中的getters属性来定义，推荐使用箭头函数，它会接收state作为其中的第一个参数。大多数时候，getter仅仅依赖state，不过有时候他们也可能会使用其他getter，因此，即使在使用常规函数定义getter时，我们也可以通过this访问到整个store实例，在typescript中必须定义返回类型。

```js
export const useStore = defineStore('main', {
  state: () => ({
    count: 0,
  }),
  getters: {
    doubleCount(state){
      return state.count * 2
    },
    // 明确设置返回值类型为number
    doublePlusOne(): number {
    	return this.doubleCount + 1
  	},
  }
})
```

在使用store的组件中

```vue
<script setup>
	import {useCounterStore} from './counterStore'
  const store = useCounterStore()
</script>
<template>
	<p>
    Double count is {{ store.doubleCount }}
  </p>
</template>
```

### 访问其他getter

与计算实行一样，我们也可以组合多个getter，通过this关键字，我们可以访问到当前定义的数据仓库中其他任何getter

### 向getter传递参数

getter只是计算属性，所以不能向他们传递任何参数，但是我们可以从getter返回一个函数，该函数能接收任意参数    

## Action

action相当于组件的method，他们可以通过defineStore中的actions属性来定义，也是定义业务逻辑的一个选择。

类似于getter，action也可以通过this访问整个store实例，并且支持完整的类型标注，不同的是action可以时异步的，我们可以在里面使用await关键字调用任何api，以及其他的action。

```js
export const useUsers = defineStore('users', {
  state: () => ({
    userData: null
  }),
  actions: {
    async registerUser(login, password) {
      try {
        this.userData = await api.post({login, password});
        showTooltip(`welcome ${this.userData.name}`)
      }catch(error){
        return error
      }
    }
  }
})
```

action 可以像函数或者通常意义上的方法一样被调用

```vue
<script setup>
const store = useCounterStore()
store.randomiszeCounter()
</script>
<template>
	<button @click='store.randomizeCounter()'></button>
</template>
```



## 插件

