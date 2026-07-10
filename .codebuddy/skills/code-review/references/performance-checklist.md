# 性能检查清单（Performance Checklist）

适用于 dehaze-system 代码评审的性能维度。覆盖前端（Vue3 SPA）、后端（Go/Python）、数据库与缓存等层面。

---

## 前端性能

### 不必要的重渲染

- [ ] 组件是否在父级状态变更时被无谓重渲染（无关 props 变化）
- [ ] `v-for` 列表是否绑定稳定的 `:key`（避免 index 作为 key）
- [ ] 计算属性是否有正确的依赖收集（避免在 computed 内使用 `Math.random()` 或时间戳）
- [ ] 大型列表是否使用虚拟滚动（数据量 > 500 条考虑 `vue-virtual-scroller`）

### Vue3 响应式陷阱

- [ ] `reactive()` 对象是否存在解构丢失响应性（应用 `toRefs` 或改用 `ref`）
- [ ] `watch` 监听是否遗漏依赖（对象/数组需 `deep: true` 或监听具体字段）
- [ ] `watchEffect` 是否因依赖收集范围过大导致频繁触发

### Pinia Store 粒度

- [ ] 是否把整个模块数据都放入一个大 store，导致无关组件因 state 变化重渲染
- [ ] action 中是否存在每次调用都发起重复接口请求（应缓存已加载数据）
- [ ] store 的 getter 是否依赖大量字段导致无意义的缓存失效

### 请求优化

- [ ] 同一接口是否存在组件挂载时的并发重复请求（多个组件同时 `onMounted` 调相同接口）
- [ ] 列表页筛选/搜索是否有防抖（建议 300ms，避免每次输入都触发接口）
- [ ] 是否存在轮询（WebSocket/SSE 场景确认是否可替代轮询）

### 图片与静态资源

- [ ] 图片是否指定合适的显示尺寸（避免加载超大图后 CSS 缩小）
- [ ] 列表中的图片是否配置懒加载（`loading="lazy"` 或 Intersection Observer）
- [ ] 大型静态资源是否通过 CDN 分发（MinIO/对象存储路径是否经过 CDN 代理）

---

## 后端性能（Go / Python 通用）

### 数据库查询

- [ ] 分页查询是否同时包含 `COUNT` 和数据查询（两次查询，避免全表 COUNT）
- [ ] 列表查询是否使用 `SELECT *`（应只 SELECT 需要的字段）
- [ ] WHERE 子句字段是否有对应索引（新增查询条件时检查 `EXPLAIN`）
- [ ] 循环内是否存在数据库查询（N+1 问题，改为批量查询或 JOIN）
- [ ] 批量插入是否使用 `INSERT INTO ... VALUES (...),(...)` 而非循环单条插入

### 缓存

- [ ] 高频读取的配置/字典数据是否走缓存（Redis），而不是每次查库
- [ ] 缓存 Key 是否设置合理 TTL（避免永不过期的热数据占用内存）
- [ ] 缓存更新时是否存在缓存击穿风险（高并发场景需加锁或 singleflight）

### 异步与并发

- [ ] Go：goroutine 是否有 context 传递，能响应取消信号
- [ ] Python：async 函数内是否有阻塞调用（`time.sleep`、同步 I/O），应改为 `asyncio.sleep` / 异步库
- [ ] 长耗时任务（去雾算法、批量处理）是否异步化，避免阻塞 HTTP handler

---

## 运行时性能

- [ ] 内存泄漏：Vue 组件 `onUnmounted` 是否清理定时器、事件监听、WebSocket 连接
- [ ] Python：异步生成器或流式读取大文件时，是否避免全量加载进内存
- [ ] Go：`defer` 在循环内部时是否会导致资源延迟释放（应提取循环体为函数）
