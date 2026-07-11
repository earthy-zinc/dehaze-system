# dehaze-java 后端模块完成情况检查报告

> 对照设计文档：`E:\DehazeSystem\dehaze-doc\docs\03-模块设计`
> 检查日期：2026-07-11

---

## 一、总体概览

| 分类 | 模块 | 完成度 | 状态 |
|------|------|--------|------|
| 基础模块 | 认证管理 | 🟢 100% | 全部完成 |
| 基础模块 | 用户管理 | 🟢 100% | 全部完成 |
| 基础模块 | 角色管理 | 🟢 100% | 全部完成 |
| 基础模块 | 菜单管理 | 🟢 100% | 全部完成 |
| 基础模块 | 部门管理 | 🟢 100% | 全部完成 |
| 基础模块 | 字典管理 | 🟢 100% | 全部完成 |
| 基础模块 | 文件管理 | 🟢 ~95% | 基本完成，有小差异 |
| 基础模块 | 任务管理 | 🟢 ~95% | 基本完成，有小差异 |
| 核心模块 | 数据集管理 | 🟢 100% | 全部完成 |
| 核心模块 | 算法管理 | 🟢 ~90% | 已补全状态/审核/版本/导入导出/监控/预测/评估 |
| 核心模块 | 图像输入 | 🟢 ~90% | 历史记录8端点已实现，配额管理已实现 |
| 核心模块 | 去雾处理 | 🟢 ~80% | 预测API已暴露，Python服务调用待联调 |
| 核心模块 | 算法选择 | 🟡 ~50% | 基础查询有，无智能推荐 |
| 核心模块 | 效果对比 | 🟢 ~75% | 评估API已暴露，对比UI属前端 |

---

## 二、逐模块详细对比

### 2.1 部门管理 — 🟢 全部完成

| 设计端点 | 实际端点 | 状态 |
|----------|----------|------|
| `GET /api/v1/dept` | `GET /api/v1/depts` | ✅ (路径差异 `dept` → `depts`) |
| `POST /api/v1/dept` | `POST /api/v1/depts` | ✅ |
| `GET /api/v1/dept/{deptId}/form` | `GET /api/v1/depts/{deptId}/form` | ✅ |
| `PUT /api/v1/dept/{deptId}` | `PUT /api/v1/depts/{deptId}` | ✅ |
| `DELETE /api/v1/dept/{ids}` | `DELETE /api/v1/depts/{ids}` | ✅ |
| `GET /api/v1/dept/options` | `GET /api/v1/depts/options` | ✅ |

**实体/服务**: `SysDept.java` / `SysDeptService` / `SysDeptMapper` ✅
**权限校验**: `sys:dept:add/edit/delete` 已配置 ✅

---

### 2.2 菜单管理 — 🟢 全部完成

| 设计端点 | 实际端点 | 状态 |
|----------|----------|------|
| `GET /api/v1/menus` | ✅ | ✅ |
| `POST /api/v1/menus` | ✅ | ✅ |
| `GET /api/v1/menus/{id}/form` | ✅ | ✅ |
| `PUT /api/v1/menus/{id}` | ✅ | ✅ |
| `DELETE /api/v1/menus/{id}` | ✅ | ✅ |
| `PATCH /api/v1/menus/{menuId}` | ✅ | ✅ |
| `GET /api/v1/menus/options` | ✅ | ✅ |
| `GET /api/v1/menus/routes` | ✅ | ✅ |

**实体/服务**: `SysMenu.java` / `SysMenuService` / `SysMenuMapper` ✅
**权限校验**: `sys:menu:add/edit/delete` 已配置 ✅

---

### 2.3 角色管理 — 🟢 全部完成

设计9个端点全部实现：
`page` / `options` / `POST` / `{roleId}/form` / `{id}` PUT / `{ids}` DELETE / `{roleId}/status` / `{roleId}/menuIds` / `{roleId}/menus` ✅

**实体/服务**: `SysRole.java` / `SysRoleMenu.java` / `SysRoleService` / `SysRoleMenuService` / `SysRoleMapper` / `SysRoleMenuMapper` ✅
**权限校验**: `sys:role:add/edit/delete` 已配置 ✅

---

### 2.4 认证管理 — 🟢 全部完成

| 设计端点 | 实际端点 | 状态 |
|----------|----------|------|
| `POST /api/v1/auth/login` | ✅ | ✅ |
| `POST /api/v1/auth/logout` | ✅ | ✅ |
| `GET /api/v1/auth/captcha` | ✅ | ✅ |
| `POST /api/v1/auth/refresh` | ✅ | ✅ |
| `GET /api/v1/auth/me` | ✅ | ✅ |

**架构实现对照设计**:
- JWT Token 机制 ✅ (含签名校验、黑名单、过期机制)
- 验证码生成(CaptchaConfig) ✅
- CaptchaValidationFilter + JwtValidationFilter 过滤器链 ✅
- BCrypt 密码加密 ✅
- Redis 黑名单 `token:blacklist:{jti}` ✅
- Spring Security 权限体系 ✅

---

### 2.5 字典管理 — 🟢 全部完成

字典类型（5个端点）+ 字典数据（6个端点）全部实现 ✅
`SysDictController` 同时处理类型和数据接口，权限标识 `sys:dict:type:*` / `sys:dict:data:*` 正确配置。

---

### 2.6 文件管理 — 🟢 ~95% 基本完成

| 设计端点 | 实际实现 | 状态 |
|----------|----------|------|
| `POST /api/v1/files` | ✅ | ✅ |
| `GET /api/v1/files/download/{objectName}` | `GET /api/v1/files/download/**` | ✅ |
| `DELETE /api/v1/files/{fileId}` | `DELETE /api/v1/files?fileId=xxx` | ⚠️ 路径参数 → Query参数 |
| `GET /api/v1/files/check?md5=` | ✅ | ✅ |
| `GET /api/v1/files/page` | ✅ | ✅ |
| `GET /api/v1/files/{fileId}` | ✅ | ✅ |

**架构实现**:
- 存储策略模式：`LocalFileService` / `MinioFileService` ✅
- `FileBOFactory` / `FilePathBuilder` ✅
- MD5去重机制 ✅

**差异项**:
- 文件删除接口设计文档为路径参数 `{fileId}`，实际为 Query 参数 `?fileId=`
- 设计文档提到的 `ImageProcessor`（格式校验/宽高解析/缩略图生成）未在 FileController 中作为独立组件使用

---

### 2.7 任务管理 — 🟢 ~95% 基本完成

| 设计端点 | 实际实现 | 状态 |
|----------|----------|------|
| `POST /api/v1/tasks` | ✅ | ✅ |
| `GET /api/v1/tasks/{taskId}` | ✅ | ✅ |
| `GET /api/v1/tasks/{taskId}/download` | ✅ | ✅ |
| `POST /api/v1/tasks/{taskId}/cancel` | `DELETE /api/v1/tasks/{taskId}` | ⚠️ POST → DELETE |
| `GET /api/v1/tasks` | ✅ | ✅ |

**架构实现对照设计**:
- 策略模式 + 工厂模式 ✅ (`TaskStrategy` / `TaskStrategyFactory`)
- 四种策略：`DatasetExportStrategy` / `ItemDownloadStrategy` / `BatchDownloadStrategy` / `CustomExportStrategy` ✅
- RabbitMQ 消息队列 ✅ (`RabbitMQPublisher` / `RabbitMQConsumer`)
- 进度回调机制 (`ProgressCallback`) ✅
- 定时清理任务 (`TaskCleanupJob`) ✅

**差异项**:
- 设计文档取消接口为 `POST /cancel`，实际为 `DELETE`
- 设计文档提到 `max.concurrent.per.user` 并发限制、死信队列等，需进一步验证是否完整实现

---

### 2.8 用户管理 — 🟢 全部完成

设计11个端点全部实现：
`page` / `POST` / `{userId}/form` / `{userId}` PUT / `{ids}` DELETE / `{userId}/password` PATCH / `{userId}/status` PATCH / `me` / `template` / `_import` / `_export` ✅

**实体/服务**: `SysUser.java` / `SysUserRole.java` / `SysUserService` / `SysUserRoleService` / `SysUserMapper` / `SysUserRoleMapper` ✅
**权限校验**: `sys:user:add/edit/delete/password:reset` 全部配置 ✅
**导入导出**: EasyExcel 实现 ✅

---

### 2.9 数据集管理 — 🟢 全部完成

**数据集接口** (7个端点): 全部实现 ✅，额外实现了 `GET /children/{parentId}` 懒加载

**数据项接口** (8个端点): 全部实现 ✅
- 配对图片上传含 `PairedImageValidator` 校验 ✅
- 批量上传含文件名解析 ✅

**图片文件接口** (5个端点): 全部实现 ✅

**实体/服务**: `SysDataset.java` / `SysDatasetItem.java` / `SysItemFile.java` / `DatasetOperationService` ✅
**权限校验**: `sys:dataset:add/edit/delete` 已配置 ✅

---

### 2.10 算法管理 — 🟢 ~90% 已补全（2026-07-11 更新）

**实际控制器**: `SysAlgorithmController`（路径 `/api/v1/algorithms`）

已实现 ✅:
| 端点 | 状态 |
|------|------|
| `GET /api/v1/algorithms` — 树形列表 | ✅ |
| `GET /api/v1/algorithms/{id}` — 详情 | ✅ |
| `GET /api/v1/algorithms/options` — 下拉选项 | ✅ |
| `POST /api/v1/algorithms` — 新增 | ✅（已补充 `@PreAuthorize("sys:algorithm:add")`） |
| `PUT /api/v1/algorithms/{id}` — 修改 | ✅ |
| `DELETE /api/v1/algorithms` — 批量删除 | ✅ |
| `PUT /api/v1/algorithms/{id}/status` — 状态变更 | ✅ **新增** |
| `PUT /api/v1/algorithms/{id}/audit` — 审核（通过/驳回） | ✅ **新增** |
| `POST /api/v1/algorithms/{id}/version` — 新增版本 | ✅ **新增** |
| `GET /api/v1/algorithms/{id}/versions` — 版本历史 | ✅ **新增** |
| `POST /api/v1/algorithms/{id}/rollback` — 版本回滚 | ✅ **新增** |
| `GET /api/v1/algorithms/{id}/_export` — 单个导出 | ✅ **新增** |
| `POST /api/v1/algorithms/_export` — 批量导出 | ✅ **新增** |
| `POST /api/v1/algorithms/_import` — 导入算法包 | ✅ **新增** |
| `POST /api/v1/algorithms/_import/validate` — 导入校验 | ✅ **新增** |
| `GET /api/v1/algorithms/{id}/monitor` — 性能监控 | ✅ **新增** |
| `GET /api/v1/algorithms/{id}/monitor/stats` — 统计报表 | ✅ **新增** |

**新增实体/服务**:
- `AlgorithmStatusEnum` — 6 状态生命周期枚举 ✅ **新增**
- `SysAlgorithm` 新增字段：`version`、`auditBy`、`auditTime`、`auditRemark` ✅
- `SysAlgorithmVersion` 实体 ✅ **新增**
- `SysAlgorithmVersionMapper` / `SysAlgorithmVersionService` / `SysAlgorithmVersionServiceImpl` ✅ **新增**
- 预测/评估 API（见下文 2.12）✅ **新增**

**剩余未实现**:
- 算法导入包 ZIP 格式支持（当前为 JSON）
- 智能推荐引擎（设计归属于"算法选择"模块）

---

### 2.11 图像输入 — 🟢 ~90% 已补全

**设计要求**:
- 图片上传：复用数据集管理 ✅（实际可用）
- 样例图片库：复用数据集查询 ✅（实际可用）
- **历史记录管理**：8个新端点 ✅（2026-07-11 已实现）

**已实现端点**:
| 端点 | 功能 | 状态 |
|------|------|------|
| `GET /api/v1/image-input/history` | 分页查询历史 | ✅ |
| `GET /api/v1/image-input/history/{id}` | 历史详情 | ✅ |
| `POST /api/v1/image-input/history` | 创建历史 | ✅ |
| `PUT /api/v1/image-input/history/{id}` | 更新历史（收藏） | ✅ |
| `DELETE /api/v1/image-input/history/{id}` | 单条删除 | ✅ |
| `DELETE /api/v1/image-input/history/batch` | 批量删除 | ✅ |
| `DELETE /api/v1/image-input/history/clear` | 清空历史 | ✅ |
| `POST /api/v1/image-input/history/sync` | 同步本地与云端 | ✅ |

- ✅ `ImageInputController` 已创建
- ✅ `SysInputHistory` 实体已创建（含 userId/algorithmId/isFavorite/syncStatus 等字段）
- ✅ `SysInputHistoryService` + `SysInputHistoryServiceImpl` 已实现
- ✅ 配额管理（默认100条，超限自动清理最旧非收藏记录）
- ✅ 用户隔离（所有操作校验 userId 归属）
- ✅ 同步机制（基础实现，标记未同步记录为已同步）

---

### 2.12 去雾处理、算法选择、效果对比 — 🟢 已补全 API 层（2026-07-11 更新）

**现状**:
- `ImageProcessingServiceImpl` 存在 ✅ — 核心去雾处理服务存在
- `SysPredLog` / `SysEvalLog` 实体 ✅
- Python 算法服务配置存在 ✅
- **PredictionController** ✅ **新增**: `POST /api/v1/prediction` / `GET /{taskId}` / `GET /logs`
- **EvaluationController** ✅ **新增**: `POST /api/v1/evaluation` / `GET /{taskId}` / `GET /logs`
- `SysPredLogService.predict()` / `getPredLogPage()` ✅ **新增**
- `SysEvalLogService.evaluate()` / `getEvalLogPage()` ✅ **新增**
- 预测/评估已对接 Python 算法服务：Java `PythonAlgorithmClient` → Python `prediction_service.py`（实际调用 `dehaze()`）/ `evaluation.py`（实际计算 PSNR/SSIM/LPIPS/NIQE），端到端联调待验证
- "效果对比"6种对比模式 属前端功能，后端提供评估数据支持 ✅
- "算法选择"智能推荐 无实现（属高级特性）

---

## 三、关键差异汇总

| # | 差异项 | 设计 | 实际 | 建议 |
|---|--------|------|------|------|
| 1 | 部门路径 | `/api/v1/dept` | `/api/v1/depts` | 统一为设计路径或更新文档 |
| 2 | 算法路径 | `/api/v1/algorithm` | `/api/v1/algorithms` | 同上 |
| 3 | 文件删除 | `DELETE /{fileId}` | `DELETE ?fileId=` | 改为 RESTful 风格 `/{fileId}` |
| 4 | 任务取消 | `POST /cancel` | `DELETE` | 改为 POST（有副作用不应幂等） |
| 5 | 算法增加**缺少权限注解** | `@PreAuthorize("sys:algorithm:add")` | 已补充 | ✅ 已修复 |
| 6 | 算法状态/审核/版本管理 | 设计有 | 已实现 | ✅ 已修复 |
| 7 | 预测/评估 API | 设计有 | 已实现 PredictionController + EvaluationController | ✅ 已修复 |
| 8 | 图像输入历史管理 | 设计有 | 已实现 | ✅ 已修复 |
| 9 | 算法版本表 | `sys_algorithm_version` | 已创建 | ✅ 已修复 |

---

## 四、建议优先级

### 🔴 P0 — 核心功能缺失（已在 2026-07-11 全部实现 ✅）
1. ✅ **模型预测 API** — 已完成 `PredictionController`
2. ✅ **效果评估 API** — 已完成 `EvaluationController`
3. ✅ **算法状态机** — 已完成 `AlgorithmStatusEnum` + 状态流转/审核/发布

### 🟡 P1 — 重要功能缺失
4. ✅ **算法版本管理** — 已完成 `SysAlgorithmVersion` + 版本创建/历史/回滚
5. ✅ **算法导入/导出** — 已完成 JSON 格式导入导出
6. ✅ **图像输入历史记录** — 已完成 8 端点 + 配额管理 + 用户隔离

### 🟢 P2 — 改善性调整
7. ✅ RESTful 风格统一（算法Controller路径参数已补全，路径仍为 `/api/v1/algorithms` vs 设计 `/api/v1/algorithm`）
8. ✅ 算法新增接口添加权限注解（已完成）
9. ✅ 监控统计 API（已完成）

---

**结论**（2026-07-11 更新）: 基础模块（8个）全部完成度高（95%~100%），核心模块中**数据集管理**、**算法管理**（含预测/评估API）和**图像输入**（含历史记录）均已完成（80%~100%）。**算法选择**智能推荐为高级特性暂未实现。预测/评估已打通 Java→Python 调用链，剩余主要待办为端到端联调验证（真实模型文件 + `dehaze()` 输出校验）。
