# 技术规范评审清单

## 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件 | camelCase | `quotationReport.ts` |
| 组件 | PascalCase | `CustomerList.tsx` |
| 目录 | kebab-case | `command-platform/` |
| 常量 | UPPER_SNAKE | `MAX_RETRY_COUNT` |
| 函数 | camelCase | `getUserInfo()` |
| Hook | use 前缀 | `useIncomePermission` |

## 类型安全

- 禁止隐式 any
- 接口/类型定义完整
- 泛型约束合理
- 类型守卫正确使用

## 状态管理

- Atom key 全局唯一
- 状态初始化在组件顶层
- 派生状态使用 Selector
- 避免单个 Atom 存储过多数据
- 优先级：`Recoil > Context > Props`

## 服务层规范

- 服务类继承 BaseService
- API 配置采用声明式定义
- 接口状态有明确标注

## 异步处理

- async 函数有 try-catch
- useEffect 中的异步有取消机制
- 错误有适当处理
- 关键操作有重试机制

## React 规范

- 组件使用 `FC<Props>` 定义
- 避免在 render 中定义函数
- 合理使用 useMemo/useCallback
- key 使用稳定且唯一的值

## 常见技术问题

1. **类型逃逸**：`as any` 绕过类型检查
2. **内存泄漏**：未清理的定时器/订阅
3. **竞态条件**：未处理组件卸载场景
4. **性能问题**：不必要的重渲染
