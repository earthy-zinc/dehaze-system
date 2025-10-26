# Vue Router

## 组合式API

### 路由访问

在setup中无法使用this访问`this.$router`或`this.$route`，应使用useRouter和useRoute函数：

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router'
const router = useRouter();
const route = useRoute();
</script>
```

route对象是响应式的，所有属性都可被监听，但应避免监听整个route对象，建议直接监听期望改变的参数。

在模板中仍可访问`$route`和`$router`。

### 导航守卫

组合式API提供导航守卫用于处理路由更新和离开页面的行为：

```vue
<script setup>
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'

onBeforeRouteLeave((to, from) => {
  const answer = window.confirm("确认离开页面吗？");
  if (!answer) return false;
})

onBeforeRouteUpdate(async (to, from) => {
  // 路由更新时的操作
})
</script>
```

组合式API守卫可用于任何由`<router-view>`渲染的组件，不局限于路由组件。

### useLink组合式函数

Vue Router将RouterLink的内部行为作为组合式函数公开，接收类似RouterLink所有prop的响应式对象，并暴露底层属性。

## 路由创建

通过createRouter函数创建路由实例，配置路由映射关系。