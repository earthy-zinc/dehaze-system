# dehaze-front-vue 测试文档

本文档描述了为 dehaze-front-vue 项目编写的测试用例，遵循项目的测试规范和最佳实践。

## 测试概述

### 测试技术栈
- **测试框架**: Vitest
- **组件测试**: Vue Test Utils
- **DOM模拟**: jsdom
- **覆盖率**: v8 provider
- **类型检查**: TypeScript

### 测试命令
```bash
# 运行所有单元测试
pnpm test:unit

# 运行测试并生成覆盖率报告
pnpm test:coverage

# 运行测试并打开UI界面
pnpm test:unit:ui

# 运行特定测试文件
pnpm test:unit src/components/Upload/__tests__/SingleUpload.spec.ts
```

## 测试结构

```
src/
├── __tests__/
│   └── README.md                   # 测试文档
├── components/
│   ├── Hamburger/
│   │   └── __tests__/
│   │       └── index.spec.ts       # 汉堡菜单组件测试
│   ├── Loading/
│   │   └── __tests__/
│   │       └── index.spec.ts       # 加载组件测试
│   ├── Magnifier/
│   │   └── __tests__/
│   │       └── index.spec.ts       # 放大镜组件测试（已存在）
│   ├── Pagination/
│   │   └── __tests__/
│   │       └── index.spec.ts       # 分页组件测试
│   ├── Upload/
│   │   └── __tests__/
│   │       └── SingleUpload.spec.ts # 单文件上传组件测试
│   └── Waterfall/
│       └── __tests__/              # 瀑布流组件测试（已存在）
├── store/modules/
│   └── __tests__/
│       ├── algorithm.spec.ts       # 算法模块Store测试
│       ├── app.spec.ts             # 应用模块Store测试
│       ├── settings.spec.ts        # 设置模块Store测试
│       └── user.spec.ts            # 用户模块Store测试（已存在）
└── utils/
    └── __tests__/
        ├── i18n.spec.ts            # 国际化工具测试
        └── index.spec.ts           # 通用工具函数测试
```

## 测试覆盖率目标

- **行覆盖率**: ≥ 80%
- **函数覆盖率**: ≥ 80%
- **分支覆盖率**: ≥ 80%
- **语句覆盖率**: ≥ 80%

## 测试规范

### 1. 测试文件命名
- 组件测试: `__tests__/index.spec.ts` 或 `__tests__/<ComponentName>.spec.ts`
- 工具函数测试: `__tests__/index.spec.ts`
- Store测试: `__tests__/<moduleName>.spec.ts`

### 2. 测试组织结构
```typescript
describe("被测试模块/组件名称", () => {
  describe("功能分组1", () => {
    it("应该测试某个具体功能点", () => {
      // 测试逻辑
    });
  });

  describe("功能分组2", () => {
    it("应该测试另一个功能点", () => {
      // 测试逻辑
    });
  });
});
```

### 3. 测试命名规范
- 使用"应该"描述期望行为
- 明确说明测试目的
- 一个测试用例只测试一个功能点

## 已编写的测试用例

### 1. 组件测试

#### Upload 组件 (`SingleUpload.spec.ts`)
- **组件渲染**: 验证组件正确渲染、显示自定义文本、图片显示
- **文件上传验证**: 文件大小限制、边界值测试
- **文件上传功能**: 成功上传、失败处理、网络错误
- **双向绑定**: modelValue响应式更新
- **Props验证**: 默认值、自定义值
- **边界情况**: 空文件、超大文件、无效类型

#### Loading 组件 (`index.spec.ts`)
- **组件渲染**: 默认状态、自定义文本验证
- **加载动画**: 动画元素数量、顺序验证
- **Props验证**: loadingText属性测试
- **响应式更新**: 动态文本更新
- **DOM结构**: 嵌套结构验证
- **边界情况**: 超长文本、特殊字符、HTML转义

#### Pagination 组件 (`index.spec.ts`)
- **组件渲染**: 默认props、自定义props验证
- **双向绑定**: page和limit的双向绑定
- **事件处理**: 页码改变、每页条数改变事件
- **Element Plus集成**: 属性传递、事件监听
- **边界情况**: total为0、超大数值、页码超出范围
- **响应式更新**: props动态更新

#### Hamburger 组件 (`index.spec.ts`)
- **组件渲染**: 正确渲染、CSS类应用
- **事件处理**: 点击事件触发、多次点击
- **Props验证**: isActive必需性、布尔类型验证
- **响应式更新**: 状态切换、类名动态更新
- **可访问性**: 键盘交互支持
- **边界情况**: 快速点击、状态快速切换

### 2. Store 模块测试

#### App Store (`app.spec.ts`)
- **初始化状态**: 默认设备类型、尺寸、语言、侧边栏状态
- **locale计算属性**: 中英文语言包切换
- **侧边栏操作**: 切换、打开、关闭功能
- **设备类型**: 设备类型切换
- **尺寸和语言**: 界面尺寸和语言切换
- **边界情况**: 无效值处理、组合操作

#### Settings Store (`settings.spec.ts`)
- **初始化状态**: 默认设置值验证
- **设置修改**: 各项设置的动态修改
- **主题切换**: 亮色/暗色主题切换、DOM更新
- **主题颜色**: 颜色变化、CSS变量更新
- **布局模式**: 布局模式切换
- **监听器**: 主题和颜色变化的DOM更新
- **边界情况**: 无效值、空值处理

#### Algorithm Store (`algorithm.spec.ts`)
- **初始化状态**: 空列表状态
- **获取数据**: 算法列表、选项列表获取
- **CRUD操作**: 添加、更新、删除算法
- **错误处理**: API调用失败、网络错误
- **边界情况**: 空参数、无效ID、不存在数据
- **完整工作流**: 完整的CRUD操作流程

### 3. 工具函数测试

#### 通用工具函数 (`index.spec.ts`)
- **DOM类操作**: hasClass、addClass、removeClass
- **CSS样式**: prefixStyle浏览器兼容性
- **环境检测**: inBrowser、IntersectionObserver支持
- **数据获取**: getValue嵌套对象访问
- **函数工具**: debounce防抖功能
- **类型检查**: isObject、isPrimitive
- **对象操作**: assign深度合并
- **URL处理**: isExternal外部链接判断
- **颜色转换**: hexToRGBA十六进制转换
- **图片加载**: loadImage异步加载
- **边界情况**: null/undefined处理、错误情况

#### 国际化工具 (`i18n.spec.ts`)
- **标题翻译**: 存在/不存在国际化配置
- **特殊情况**: 空字符串、特殊字符、数字标题
- **API交互**: i18n方法调用验证
- **边界情况**: 超长标题、特殊符号、emoji
- **性能优化**: 方法调用次数验证
- **类型安全**: 各种输入类型处理

## 测试策略

### 1. 组件测试策略
- **渲染测试**: 验证组件正确渲染
- **Props测试**: 验证属性传递和默认值
- **事件测试**: 验证事件触发和参数
- **状态测试**: 验证组件内部状态管理
- **交互测试**: 验证用户交互行为
- **边界测试**: 验证极端情况处理

### 2. Store 测试策略
- **状态初始化**: 验证初始状态正确性
- **Action测试**: 验证状态变更操作
- **Getter测试**: 验证计算属性正确性
- **异步操作**: 验证API调用和错误处理
- **状态持久化**: 验证localStorage集成

### 3. 工具函数测试策略
- **功能正确性**: 验证函数基本功能
- **边界情况**: 验证极端输入处理
- **错误处理**: 验证异常情况处理
- **性能考虑**: 验证函数调用效率
- **兼容性**: 验证浏览器兼容性

## Mock 策略

### 1. 外部依赖Mock
```typescript
// Mock SDK
vi.mock("dehaze-sdk-js", () => ({
  FileAPI: { upload: vi.fn() },
  AlgorithmAPI: { getList: vi.fn() },
}));

// Mock Element Plus
vi.mock("element-plus", async () => ({
  ElMessage: { warning: vi.fn() },
}));
```

### 2. DOM API Mock
```typescript
// Mock window and document
Object.defineProperty(window, "location", {
  value: { host: "localhost:5174" },
  writable: true,
});

// Mock Image constructor
global.Image = vi.fn(() => ({ src: "", onload: null }));
```

### 3. 时间Mock
```typescript
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});
```

## 运行测试

### 本地开发
```bash
# 安装依赖
pnpm install

# 运行测试
pnpm test:unit

# 生成覆盖率报告
pnpm test:coverage
```

### CI/CD 集成
测试已配置为在以下情况下自动运行：
- 提交代码前（通过 husky）
- Pull Request 创建时
- 定时任务（每日）

## 贡献指南

### 添加新测试
1. 在对应目录创建 `__tests__` 文件夹
2. 编写符合规范的测试文件
3. 确保测试覆盖率达到80%以上
4. 运行测试确保通过

### 测试审查清单
- [ ] 测试命名清晰明确
- [ ] 每个测试只验证一个功能点
- [ ] 包含成功和失败场景
- [ ] 包含边界情况测试
- [ ] Mock外部依赖
- [ ] 清理测试状态
- [ ] 覆盖率达到要求

## 常见问题

### Q: 如何测试异步操作？
A: 使用 `async/await` 或 `Promise.then()` 处理异步操作，使用 `vi.useFakeTimers()` 模拟时间。

### Q: 如何Mock第三方库？
A: 使用 `vi.mock()` 在测试文件顶部Mock第三方库。

### Q: 如何测试组件的DOM操作？
A: 使用 Vue Test Utils 提供的 `find()`, `trigger()` 等方法。

### Q: 如何验证覆盖率？
A: 运行 `pnpm test:coverage` 查看详细覆盖率报告。

---

最后更新: 2025-11-01
版本: 1.0.0