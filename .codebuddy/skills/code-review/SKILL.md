---
name: code-review
description: 业务感知型代码评审技能。当用户提及"代码评审"、"code review"、"review代码"、"评审PR"、"优化代码"、"简化代码"、"代码清理"，或要求检查代码是否符合业务需求、评估代码设计合理性时，触发此技能。结合需求文档和项目规范，从业务契合度、架构设计、技术规范、性能优化、代码简化五个维度进行深度评审，生成结构化评审报告。
---

# AI Code Review Skill

ESLint/Prettier 管格式，AI 管业务逻辑和架构设计。

## Getting Started

根据用户输入确定评审范围：

- **指定文件**：直接读取
- **需求文档**：提取需求关键点，用于业务层评审
- **PR/分支评审**：执行 `git diff --stat <base>...HEAD` 和 `git diff --name-only <base>...HEAD` 获取变更概览

对每个变更文件，识别所属模块（用户/数据集/任务等），使用 `read_rules` 工具加载对应的项目编码规范（如 `go-coding.md`、`python-coding.md`、`frontend-coding.md` 等）。

如果当前需求有对应的代码设计文档（通常位于 `dehaze-doc/docs/项目文档/图像去雾系统/新结构/03-模块设计/[模块]/`），应优先读取并作为业务层和架构层评审的参考依据。对照设计文档验证实现是否偏离预期设计，将偏差作为评审发现项记录。

## Pre-Review: Lint Auto-Fix

在正式评审前，建议先运行 lint 工具自动修复基础问题：

```bash
# 在 dehaze-front-vue 目录下执行
cd dehaze-front-vue
pnpm lint

# Go 后端（如需要）
cd dehaze-go
go fmt ./...
go vet ./...

# Python 后端（如需要）
cd dehaze-python
ruff check --fix .
```

## Core Workflows

按五个维度逐一评审，各维度的详细检查点参考：

- 业务层：`references/business-checklist.md` — 需求覆盖、边界处理、流程完整性
- 架构层：`references/architecture-checklist.md` — 模块职责、依赖方向、代码复用
- 技术层：`references/technical-checklist.md` — 类型安全、状态管理、异步处理
- 性能层：`references/performance-checklist.md` — 数据库与缓存、运行时性能、前端优化
- 简化层：`references/simplification-checklist.md` — 命名可读性、函数设计、代码重复、控制流

### 严重程度标识

| 级别 | 含义 |
|------|------|
| 阻塞 | 必须修复，影响正确性或安全 |
| 警告 | 建议修复，影响可维护性 |
| 优化 | 可以改进，提升性能或可读性 |
| 通过 | 符合规范 |
| 亮点 | 优秀实践，值得推广 |

### 评审原则

1. 不重复 ESLint 工作，聚焦业务和设计
2. 指出问题同时给出改进建议和代码示例
3. 结合上下文，避免脱离实际的建议
4. 代码简化以保持功能不变为前提，只改实现方式不改行为
5. 选择清晰而非简洁——显式代码通常优于过度紧凑的代码

## Auto-Fix Mode

当用户要求"自动修复"或类似请求时：

1. 优先修复「阻塞」和「警告」级别的问题
2. 对「优化」级别的问题提供修复选项，不擅自改动
3. 每次修改后验证可编译/可运行
4. 保留原代码备份或 git commit 点

## Output Format

参考以下结构输出五维度评审报告：

# Code Review Report
> 分支: feature/xxx

## 概览
- 变更文件数: X
- 业务契合度: X/5
- 架构设计: X/5
- 代码质量: X/5
- 性能表现: X/5
- 综合评分: X/5

## 业务层评审
- [阻塞] 缺少空数据状态展示
  - 位置: xxx.ts:45
  - 建议: 使用 <EmptyState /> 组件包裹

## 架构层评审
- [通过] 模块职责清晰，组件粒度合理

## 技术层评审
- [警告] useEffect 中的异步请求未设置取消机制
  - 位置: xxx.ts:20-35
  - 建议: 添加 AbortController

## 性能层评审
- [优化] 大列表未使用虚拟滚动
  - 位置: xxx.vue:88
  - 建议: 使用 VirtualList 组件或分页

## 简化层评审
- [优化] 魔术数字未提取为常量
  - 位置: xxx.py:120
  - 建议: 定义常量 `MAX_RETRY_COUNT = 3`

## 改进建议
1. [高] 补充空数据状态处理
2. [中] 异步请求添加取消机制
3. [低] 提取常量

## 亮点
- 列表筛选逻辑抽取为独立 hook，复用性好
- 使用了合理的 TypeScript 类型定义

## 评审结论
结论: 有条件通过 - 修复阻塞问题后可合并

## Examples

**输入**：用户说「帮我 review 一下 UserList.vue」

**输出结构**：

```
# Code Review Report
> 分支: feature/user-list
> 文件: dehaze-front-vue/src/views/user/UserList.vue

## 概览
- 变更文件: UserList.vue
- 业务契合度: 4/5
- 架构设计: 3/5
- 代码质量: 3/5
- 性能表现: 3/5
- 综合评分: 3.25/5

## 业务层评审
- [警告] 禁用超级管理员（root）的权限校验未实现
  - 位置: UserList.vue:145
  - 建议: 在删除接口调用前检查 `row.username === 'root'`
- [通过] 用户列表分页、筛选功能完整

## 架构层评审
- [警告] 用户列表组件与用户详情组件耦合度较高
  - 位置: UserList.vue:50-75
  - 建议: 将详情抽取为独立组件或抽离到 store

## 技术层评审
- [警告] 未对 username 进行小写转换可能导致查询不一致
  - 位置: UserList.vue:89
  - 建议: 调用接口前统一 `username.toLowerCase()`
- [通过] Pinia store 使用正确

## 性能层评审
- [优化] 用户数据量大时列表渲染可能卡顿
  - 位置: UserList.vue:120
  - 建议: 考虑使用虚拟滚动或增加分页大小
- [优化] 请求未做防抖处理
  - 位置: UserList.vue:78
  - 建议: 搜索输入框添加防抖

## 简化层评审
- [优化] 过滤逻辑重复，可提取为 computed
  - 位置: UserList.vue:156-167
  - 建议: 定义 `filteredUsers` computed

## 改进建议
1. [高] 补充 root 用户删除防护
2. [高] 统一用户名小写转换
3. [中] 分离详情组件或使用抽离方案
4. [低] 提取过滤逻辑为 computed
5. [低] 搜索输入添加防抖

## 亮点
- 组件结构清晰，注释完整
- 使用了 TypeScript 类型定义

## 评审结论
结论: 有条件通过 - 修复高优先级问题后可合并
```
