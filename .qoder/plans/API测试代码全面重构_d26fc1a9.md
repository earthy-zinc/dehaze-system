# API 测试代码全面重构

## 背景分析

现有测试文件共 12 个模块：`dept`、`user`、`role`、`dict`、`menu`、`dataset`、`algorithm`、`file`、`model`、`image-input/history`，外加 `factories` 数据工厂和 `utils/assertion.ts` 断言工具。

**主要问题汇总：**
1. 缺少独立的安全性测试文件
2. `role.test.ts`、`dict.test.ts`、`menu.test.ts` 中大量使用 `if (list.length === 0) return` 静默跳过
3. `dataset.test.ts`、`algorithm.test.ts` 存在弱断言（`toBeDefined()` 不验证值）
4. `history.test.ts` 详情查询断言不够完整
5. 数据清理逻辑分散在每个文件中，代码重复
6. 缺少完整的 CRUD 生命周期闭环验证（创建→读→更新→读→删→验证不存在）
7. 缺少安全性测试（XSS 注入、SQL 注入、超长字符串、401/403）

## 任务清单

### Task 1: 创建统一的测试清理工具

**文件：** `test/utils/cleanup.ts`

- 创建 `TestCleanupRegistry` 类，统一注册/执行清理操作
- 支持按模块注册清理回调，自动在 `afterAll` 中按后进先出顺序执行
- 忽略清理过程中的错误，确保所有清理都被尝试执行
- 替代现有各文件中重复的 `for (id of ids) { try { await API.delete(id) } catch {} }` 模式

```typescript
// 示例用法
const cleanup = new TestCleanupRegistry();
afterAll(() => cleanup.executeAll());
// 在测试中注册
cleanup.register(() => DeptAPI.deleteByIds(deptId.toString()));
```

---

### Task 2: 创建安全性测试套件

**文件：** `test/modules/security/security.test.ts`

参考设计文档安全性测试要求，补充以下测试：

**2.1 输入安全测试：**
- XSS 脚本注入测试（部门、用户、角色、字典、数据集名称字段）：`<script>alert("xss")</script>` → 验证后端拒绝或转义
- SQL 注入测试：`' OR '1'='1` / `admin'--` → 验证不会泄露数据
- 超长字符串拦截：name 字段 10000 字符 → 验证后端返回参数校验错误

**2.2 认证安全测试（若 SDK 支持）：**
- 未登录访问受保护接口（调用 logout 后访问 DeptAPI/UserAPI 等）→ 期望抛出 401 类错误

**2.3 边界防护测试：**
- 对 name/description 等字符串字段传入特殊字符（`<>&"'`）验证不产生存储污染

---

### Task 3: 改进 dept.test.ts（部门管理）

**重点改进：**
- 移除 `beforeAll` 中的历史脏数据清理逻辑（应属于迁移脚本而非测试）
- 将 `POST` 测试中的"参数校验：缺少 name"从 `expectBizErrorOrUndefined` 改为明确断言（若后端未校验则保留为已知问题标注）
- 补充完整 CRUD 生命周期测试：创建→读取验证字段→更新→读取验证更新→删除→验证不存在
- 补充边界场景：超长名称、特殊字符名称

---

### Task 4: 改进 user.test.ts（用户管理）

**重点改进：**
- 将 `beforeAll` 中 `testUserIds` 改为每个测试独立创建数据（消除隐式依赖）
- 删除 `if (pageResult.list.length === 0) return` 静默跳过，改为 `expect` 显式断言
- 加强 `DELETE` 测试中的删除后验证（不仅验证 `getFormData` 返回空，还要验证分页列表中不存在）
- 补充安全性测试：XSS 字符注入到 nickname 字段

---

### Task 5: 改进 role.test.ts（角色管理）

**重点改进：**
- 移除所有 `if (!setupSuccess || !testRoleId) { console.log("跳过测试"); return; }` 静默跳过模式，改为 `beforeAll` 中使用 `expect` 断言创建成功，或使用 `test.skipIf` 显式跳过
- 将 `DELETE` 测试中的 `expect(true).toBe(false)` 强制失败改为使用 `expectBizError` 断言（对齐 dict.test.ts 的做法）
- 补充完整 CRUD 生命周期测试

---

### Task 6: 改进 dict.test.ts（字典管理）

**重点改进：**
- 移除所有 `if (list.length === 0) return` 静默跳过，改为 `expect(list.length).toBeGreaterThan(0)` 或在 `beforeAll` 中确保数据存在
- 补充下拉列表字段类型断言（`typeof option.value === 'string'`）
- 补充完整 CRUD 生命周期测试（字典类型 + 字典数据的联合生命周期）

---

### Task 7: 改进 menu.test.ts（菜单管理）

**重点改进：**
- 移除所有 `if (allMenus.length === 0) { console.warn(...); return; }` 静默跳过，改为显式断言
- 加强下拉列表断言：`typeof option.value`、`typeof option.label`
- 补充边界测试：超长菜单名称
- 移除 `expect(resolves.not.toThrow())` 弱断言，改为验证具体返回值

---

### Task 8: 改进 dataset.test.ts（数据集管理）

**重点改进：**
- 加强弱断言：`expect(firstItem.id).toBeDefined()` → 验证具体值或类型
- 补充完整 CRUD 生命周期测试
- 加强下拉选项断言：验证字段类型和非空
- 补充边界测试：特殊字符名称、超长描述

---

### Task 9: 改进 algorithm.test.ts（算法管理）

**重点改进：**
- 加强弱断言：下拉列表 `option.value` 验证类型
- 补充完整 CRUD 生命周期测试
- 移除 `console.warn` 静默跳过，改为显式断言
- 加强详情查询断言：验证所有字段与创建时一致

---

### Task 10: 改进 history.test.ts（图像输入历史）

**重点改进：**
- 加强详情查询断言：验证具体字段值（`originalImageUrl`、`algorithmName` 与创建时一致）
- 移除 `if (createdIds.length === 0) return` 静默跳过，改为 `beforeAll` 中确保数据存在

---

### Task 11: 补充 file.test.ts 和 model.test.ts

**file.test.ts 改进：**
- 补充文件大小/格式相关的边界测试（若 SDK 支持）

**model.test.ts 改进：**
- 将 `toHaveProperty` 弱断言改为验证字段类型和值范围

---

## 执行优先级

| 优先级 | 任务 | 原因 |
|--------|------|------|
| P0 | Task 1 (清理工具) | 后续所有任务依赖 |
| P0 | Task 2 (安全性测试) | 用户明确要求 |
| P0 | Task 3-5 (dept/user/role) | 核心模块，问题最多 |
| P1 | Task 6-7 (dict/menu) | 静默跳过问题严重 |
| P1 | Task 8-9 (dataset/algorithm) | 弱断言问题 |
| P2 | Task 10-11 (history/file/model) | 改进幅度较小 |

## 改动范围估计

- 新建文件：2 个（`cleanup.ts`、`security.test.ts`）
- 修改文件：10 个（所有 `*.test.ts`）
- 预计新增代码：约 800-1200 行
- 预计修改代码：约 200-400 行