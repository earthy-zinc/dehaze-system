# CSS 模块化架构说明

## 📁 文件结构

```
styles/
├── base.css          # 基础样式模块
├── layout.css        # 布局组件模块
├── components.css    # 通用组件模块
├── features.css      # 功能组件模块
├── dataset.css       # 数据集管理模块
├── homepage.css      # 首页样式模块
├── responsive.css    # 响应式适配模块
└── README.md         # 本文档
```

## 🎯 模块职责划分

### 1. base.css - 基础样式模块
**职责**：提供全局基础样式和工具类
- 全局样式重置（`*`, `body`, `html`）
- 通用动画定义（`@keyframes`）
- 工具类（`.line-clamp-*`, `.aspect-square`）
- 滚动条样式
- 触摸反馈效果

**何时修改**：
- 添加新的全局动画效果
- 定义新的工具类
- 调整全局字体或重置样式

### 2. layout.css - 布局组件模块
**职责**：页面布局相关的组件样式
- Header 头部导航
- 底部导航栏（`.nav-item`）
- 侧边菜单（`.menu-item`）
- 模态框（`.modal`）
- 网格布局（`.image-grid`）

**何时修改**：
- 调整页面整体布局
- 修改导航栏样式
- 添加新的布局容器

### 3. components.css - 通用组件模块
**职责**：可复用的UI组件样式
- 按钮（`.btn`, `.btn-primary`, `.btn-secondary`）
- 卡片（`.card`, `.feature-card`, `.metric-card`）
- 标签（`.badge`）
- 滑块（`.slider`）
- 开关（`.toggle-switch`）
- 工具栏（`.toolbar`）
- 加载动画（`.loader`）

**何时修改**：
- 添加新的通用UI组件
- 调整按钮、卡片等基础组件样式
- 优化组件交互效果

### 4. features.css - 功能组件模块
**职责**：业务功能相关的组件样式
- 图片容器（`.image-container`）
- 图片上传（`.upload-area`）
- 图片预览（`.image-preview`）
- 分割线（`.divider-line`）
- 放大镜（`.magnifier`）
- 算法选择（`.algorithm-card`）
- 滤镜控制（`.filter-control`, `.preset-btn`）
- 处理模式（`.mode-btn`）

**何时修改**：
- 添加新的图片处理功能
- 调整对比工具样式
- 优化滤镜调节界面

### 5. dataset.css - 数据集管理模块
**职责**：数据集管理功能的样式
- 数据集卡片（`.dataset-card`）
- 图片类型标签（`.type-tab`, `.type-badge`）
- 瀑布流布局（`.waterfall-grid`）
- 数据集信息（`.dataset-info-card`）
- 统计数据（`.stats-grid`, `.stat-box`）

**何时修改**：
- 调整数据集展示样式
- 优化图片网格布局
- 修改筛选按钮样式

### 6. homepage.css - 首页样式模块
**职责**：首页专用的样式
- Hero 区域（`.hero-section`）
- 效果展示（`.showcase-section`）
- 工作流程（`.workflow-container`, `.workflow-step`）
- 工具网格（`.tools-grid`, `.tool-card`）
- 算法优势（`.algorithm-section`）
- 技术规格（`.tech-specs-section`, `.spec-card`）
- 行动号召（`.final-cta-section`）

**何时修改**：
- 调整首页布局和设计
- 优化营销展示效果
- 修改工作流程展示

### 7. responsive.css - 响应式适配模块
**职责**：不同设备的适配样式
- 手机横屏适配
- 平板横屏/竖屏适配
- 桌面端适配
- 超大屏适配
- 数据集模块响应式

**何时修改**：
- 添加新设备尺寸支持
- 优化特定设备的显示效果
- 调整断点和布局策略

## 🔧 开发规范

### 命名规范
- 使用 BEM 方法论或语义化命名
- 类名使用小写字母和连字符（kebab-case）
- 避免使用过于通用的类名

### 样式组织
- 相关样式放在一起
- 使用注释分隔不同功能区域
- 保持代码缩进和格式一致

### 修改流程
1. 确定要修改的样式属于哪个模块
2. 在对应的模块文件中进行修改
3. 如果是新功能，考虑是否需要创建新模块
4. 测试修改在不同设备上的效果

### 添加新样式
1. **通用组件**：添加到 `components.css`
2. **业务功能**：添加到 `features.css` 或对应的功能模块
3. **响应式样式**：添加到 `responsive.css`
4. **全局工具类**：添加到 `base.css`

## 📊 模块依赖关系

```
styles.css (主文件)
    ├── base.css (基础层，无依赖)
    ├── layout.css (依赖 base.css)
    ├── components.css (依赖 base.css)
    ├── features.css (依赖 base.css, components.css)
    ├── dataset.css (依赖 base.css, components.css)
    ├── homepage.css (依赖 base.css, components.css)
    └── responsive.css (覆盖所有模块，最后加载)
```

## 🎨 样式优先级

1. **基础样式** (base.css) - 最低优先级
2. **布局样式** (layout.css)
3. **组件样式** (components.css)
4. **功能样式** (features.css, dataset.css, homepage.css)
5. **响应式样式** (responsive.css) - 最高优先级

## 🚀 性能优化建议

### 当前架构优势
- ✅ 模块化加载，便于维护
- ✅ 职责清晰，减少样式冲突
- ✅ 响应式样式集中管理
- ✅ 便于团队协作开发

### 未来优化方向
- 考虑使用 CSS 预处理器（Sass/Less）
- 实现按需加载（Critical CSS）
- 使用 PostCSS 进行自动化优化
- 考虑 CSS Modules 或 CSS-in-JS

## 📝 维护检查清单

### 添加新功能时
- [ ] 确定样式应该放在哪个模块
- [ ] 检查是否有可复用的现有样式
- [ ] 添加必要的注释说明
- [ ] 测试响应式效果
- [ ] 更新本文档（如有必要）

### 修改现有样式时
- [ ] 确认修改范围和影响
- [ ] 检查是否影响其他页面
- [ ] 测试不同设备的显示效果
- [ ] 保持代码风格一致

### 代码审查时
- [ ] 样式是否放在正确的模块
- [ ] 命名是否符合规范
- [ ] 是否有重复的样式定义
- [ ] 响应式适配是否完整

## 🔍 常见问题

### Q: 如何决定样式应该放在哪个模块？
A: 遵循以下原则：
- 通用的、可复用的 → `components.css`
- 特定业务功能的 → `features.css` 或对应功能模块
- 布局相关的 → `layout.css`
- 响应式适配的 → `responsive.css`

### Q: 可以在多个模块中定义相同的类名吗？
A: 应该避免。如果需要覆盖，应该：
1. 使用更具体的选择器
2. 在 `responsive.css` 中进行响应式覆盖
3. 考虑重构为更通用的组件

### Q: 如何处理样式冲突？
A: 
1. 检查样式的加载顺序
2. 使用更具体的选择器
3. 考虑使用 CSS 变量统一管理
4. 重构冲突的样式为独立组件

## 📚 相关资源

- [BEM 命名规范](http://getbem.com/)
- [CSS 架构最佳实践](https://developer.mozilla.org/zh-CN/docs/Learn/CSS/Building_blocks/Organizing)
- [响应式设计指南](https://web.dev/responsive-web-design-basics/)

## 📅 更新日志

### v1.0.0 (2025-01-21)
- 🎉 初始版本
- ✨ 完成 CSS 模块化重构
- 📝 创建架构文档

---

**维护者**: with AI  
**最后更新**: 2025-01-21  
**版本**: 1.0.0
