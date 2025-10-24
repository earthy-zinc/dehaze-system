---
trigger: model_decision
description: 当你对 dehaze-front-react 项目编写测试用例时，请务必参考该规则。
---

# dehaze-front-react 项目测试规范

## 1. 测试架构概述

dehaze-front-react 项目采用了完整的测试体系，包括单元测试和端到端(E2E)测试两个层面。

### 1.1 测试工具栈
- **单元测试工具**: Vitest + React Testing Library
- **端到端测试工具**: Playwright
- **测试辅助工具**: 
  - [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) - React 组件测试工具
  - [vitest-canvas-mock](https://github.com/codyzu/vitest-canvas-mock) - Canvas 相关测试模拟
  - [jsdom](https://github.com/jsdom/jsdom) - DOM 环境模拟

### 1.2 测试分类
- **单元测试**: 针对组件、函数、store 等独立单元的功能验证
- **集成测试**: 验证多个组件或模块之间的交互
- **端到端测试**: 模拟真实用户行为的全流程测试

## 2. 单元测试规范

### 2.1 测试文件组织

- 测试文件应放置在被测试组件或模块的同级目录下的 `__tests__` 文件夹中
- 测试文件命名: `[组件名].spec.tsx` 或 `[模块名].test.ts`

示例：
```
src/
  components/
    Magnifier/
      index.tsx
      __tests__/
        index.spec.tsx
```

### 2.2 测试覆盖率要求
- **最低覆盖率**: 80%
- **覆盖指标**: 行覆盖率(lines)、函数覆盖率(functions)、分支覆盖率(branches)、语句覆盖率(statements)

### 2.3 测试编写规范

#### 2.3.1 测试结构
使用 describe-it 结构组织测试套件：
```typescript
describe("组件/模块名称", () => {
  describe("功能描述", () => {
    it("应该...", () => {
      // 测试内容
    });
  });
});
```

#### 2.3.2 组件测试要点
1. **渲染测试**
   - 验证组件是否正确渲染
   - 检查根据 props 渲染不同状态
   - 验证条件渲染和列表渲染

2. **交互测试**
   - 模拟用户事件(click, input, keydown等)
   - 验证事件处理函数的调用
   - 检查状态变更和DOM更新

3. **Props测试**
   - 验证默认props值
   - 测试不同props值的行为
   - 验证props验证器

4. **Redux状态测试**
   - 验证组件正确使用useSelector获取状态
   - 验证组件正确使用useDispatch派发动作
   - 测试异步操作(createAsyncThunk)的结果

5. **异步操作测试**
   - API调用
   - 状态更新
   - 加载状态处理

### 2.4 特殊场景测试

#### 2.4.1 Canvas相关测试
- 使用 [vitest-canvas-mock]() 模拟 Canvas API
- 验证 Canvas 上下文方法调用
- 测试绘图逻辑和边界情况

#### 2.4.2 网络请求模拟
- 使用 `vi.mock` 模拟 API 调用
- 分别测试成功和失败场景
- 验证加载状态和错误处理

#### 2.4.3 Redux状态管理测试
- 使用 `redux-mock-store` 或直接测试 slice reducers
- 验证 createAsyncThunk 的 fulfilled 和 rejected 状态
- 测试 Redux Persist 的行为

#### 2.4.4 边界情况测试
- 输入验证
- 空值处理
- 极端数值处理
- 异常流程处理

## 3. 端到端测试规范

### 3.1 测试文件组织
- 端到端测试文件位于项目根目录的 [e2e](../../dehaze-front-react/e2e) 目录下
- 文件命名: `[功能].spec.ts`

### 3.2 测试重点
1. 关键用户流程验证
2. 页面间导航
3. 表单提交和验证
4. 用户身份验证流程
5. 权限相关功能
6. 图像处理功能流程

## 4. 测试执行命令

### 4.1 单元测试命令
```bash
# 运行所有单元测试
pnpm test:unit

# 运行所有单元测试并生成覆盖率报告
pnpm test:coverage
```

### 4.2 端到端测试命令
```bash
# 运行所有端到端测试
pnpm test:e2e
```

### 4.3 特定测试文件执行
对于特定组件的测试，可以使用以下命令：
```bash
npx vitest run src/components/Loading/__tests__/index.spec.tsx --coverage
```

## 5. 测试最佳实践

### 5.1 编写原则
1. **单一职责**: 每个测试只验证一个功能点
2. **独立性**: 测试之间不应相互依赖
3. **明确性**: 测试名称应清楚表达预期行为
4. **简洁性**: 避免过度复杂的测试设置

### 5.2 Mock策略
1. 外部API调用必须mock
2. Redux store在组件测试中应使用 mock store 或通过 Provider 提供真实 store
3. 时间相关函数(Date.now, setTimeout等)需要mock
4. 随机值生成需要固定种子
5. 复杂的第三方库应适当mock

### 5.3 断言原则
1. 优先使用具体值比较而非模糊匹配
2. 验证行为而不仅仅是状态
3. 避免断言过多无关细节
4. 对错误情况也要进行断言

### 5.4 性能优化
1. 合理使用 `beforeEach` 和 `afterEach`
2. 避免在测试中进行不必要的网络请求
3. 对耗时操作使用适当的超时设置
4. 重用测试设置以提高执行效率

## 6. 重要事项

请牢记以下重要事项：

- 你在测试每个组件/函数时，应阅读源文件了解其想要实现的功能，并分析该功能与其他功能之间的联系，确保功能设计是符合预期的。
- 所给出的测试应避免包含组件/函数的实现细节，而是测试其功能是否符合预期。
- React组件测试应使用 React Testing Library，关注组件的行为而不是实现细节。
- 对于涉及Canvas的组件（如Magnifier），需要正确配置 vitest-canvas-mock 环境。