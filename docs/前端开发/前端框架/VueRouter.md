# VueRouter 学习

## 组合式API

### setup中访问路由

在setup中不存在this，因此无法直接访问`this.$router`或者`this.$route`作为替代，使用useRouter和useRoute函数

```vue
<script setup>
	import {useRouter, useRoute} from 'vue-router'
  const router = useRouter();
  const route = useRoute();
</script>
```

`route`对象是一个响应式对象，因此它所有的属性都可以被监听，但是应该避免监听整个route对象，在大多数情况下，应该直接监听期望改变的参数。

在模板中仍然能够访问`$route`和`$router`

### 导航守卫

更新和离开当前页面时所采用的导航行为采用，

```vue
<script>
  import {onBeforeRouteLeave, 
          onBeforeRouteUpdate} 
  from 'vue-router'
  onBeforeRouteLeave((to, from) => {
    const answer = window.confirm("do you want it?");
    if(!answer) return false;
  })
  onBeforeRouteUpdate(async (to, from) => {
    // 操作
  })
</script>
```

组合式API守卫可以运用在任何由`<router-view>`渲染的组件中，不必向组件内守卫那样直接用在路由组件上。

### useLink

Vue Router将RouterLink的内部行为作为一个组合式函数公开，它接收一个类似RouterLink所有prop响应式对象，并且暴露底层属性。

## 创建路由

