# Pinia

## Store定义

Store是Vue的数据存储仓库。在setup中通过`defineStore()`定义，第一个参数是唯一标识符，第二个参数可以是配置对象或函数。

### 配置对象方式

配置对象可包含state、getters、actions属性：
- state：存储数据
- getters：计算属性
- actions：方法

```js
const useCounterStore = defineStore('counter', {
  state: () => ({ count: 0 }),
  getters: { 
    double: (state) => state.count * 2
  },
  actions: {
    increment() {
      this.count++
    }
  }
})
```

### 函数方式

通过函数定义响应式属性和方法：

```js
const useCounterStore = defineStore('counter', () => {
  const count = ref(0);
  function increment() {
    count.value++
  }
  return { count, increment }
})
```

函数方式与配置对象方式的对应关系：
- ref() 对应 state 属性
- computed() 对应 getter 属性
- function() 对应 action 属性

## Store使用

在组件中调用use...Store()函数获取store实例：

```vue
<script setup>
import { storeToRefs } from 'pinia'
const store = useCounterStore();
const { name, doubleCount } = storeToRefs(store);
const { increment } = store;
</script>
```

使用storeToRefs()可保持响应性，适用于仅使用状态而不调用action的场景。

## State状态管理

State是store的数据部分，定义为返回初始状态的函数：

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

### State变更

两种变更方式：
1. 直接修改：`store.count++`
2. 使用$patch方法批量修改：

```js
store.$patch({
  count: store.count + 1,
  age: 120,
  name: 'DIO'
})
```

### State重置

调用store的$reset()方法将state重置为初始值。

### State侦听

通过$subscribe()方法侦听state变化，订阅方法在patch分发后只触发一次：

```js
cartStore.$subscribe((mutation, state) => {
  mutation.type
  mutation.storeId
  mutation.payload
  localStorage.setItem('cart', JSON.stringify(state))
})
```

默认情况下，state subscription会绑定到添加它们的组件上，组件卸载时自动删除订阅。如需在组件卸载后保留订阅，需设置`{ detached: true }`参数。

## Getter计算属性

Getter相当于store的计算属性，通过defineStore中的getters属性定义：

```js
export const useStore = defineStore('main', {
  state: () => ({
    count: 0
  }),
  getters: {
    doubleCount(state) {
      return state.count * 2
    },
    // 明确设置返回值类型为number
    doublePlusOne(): number {
      return this.doubleCount + 1
    }
  }
})
```

在组件中使用getter：

```vue
<script setup>
import { useCounterStore } from './counterStore'
const store = useCounterStore()
</script>
<template>
  <p>Double count is {{ store.doubleCount }}</p>
</template>
```

### Getter组合

Getter可以像计算属性一样组合使用，通过this关键字访问其他getter。

### 参数化Getter

Getter不能直接接收参数，但可以返回接收参数的函数：

```js
export const useStore = defineStore('main', {
  getters: {
    getUserById: (state) => {
      return (userId) => state.users.find(user => user.id === userId)
    }
  }
})
```

## Action动作

Action相当于组件的方法，通过defineStore中的actions属性定义，用于实现业务逻辑。

Action可通过this访问整个store实例，支持完整的类型标注，可为异步操作：

```js
export const useUsers = defineStore('users', {
  state: () => ({
    userData: null
  }),
  actions: {
    async registerUser(login, password) {
      try {
        this.userData = await api.post({ login, password });
        showTooltip(`welcome ${this.userData.name}`)
      } catch (error) {
        return error
      }
    }
  }
})
```

调用action的方式：
1. 函数调用：`store.randomizeCounter()`
2. 事件绑定：`@click='store.randomizeCounter()'`

```vue
<script setup>
const store = useCounterStore()
store.randomizeCounter()
</script>
<template>
  <button @click='store.randomizeCounter()'></button>
</template>
```

## 插件机制

Pinia支持插件扩展功能，可通过插件添加全局属性、修改store行为等。