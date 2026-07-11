# 交接文档：dehaze-java 后端模块补全 & SDK API 测试

> 编写日期：2026-07-11
> 编写人：AI 助手
> 交接对象：后续开发人员

---

## 一、任务背景

对照设计文档 `dehaze-doc/docs/03-模块设计`，检查 dehaze-java 后端 14 个模块的完成情况，补全缺失功能，并在 `dehaze-tool/dehaze-sdk-js` 中新增对应 API 端点和测试。

---

## 二、已完成工作

### 2.1 模块完成情况检查报告

生成了完整的检查报告：`deliverables/dehaze-java-module-completion-report.md`

### 2.2 Java 后端新增代码

#### 新增文件清单（24 个）

**枚举**
- `common/enums/AlgorithmStatusEnum.java` — 算法 6 状态生命周期枚举

**实体**
- `model/entity/SysAlgorithmVersion.java` — 算法版本历史表
- `model/entity/SysInputHistory.java` — 图像输入历史记录表

**Mapper**
- `mapper/SysAlgorithmVersionMapper.java`
- `mapper/SysInputHistoryMapper.java`

**Service**
- `service/SysAlgorithmVersionService.java` + `impl/SysAlgorithmVersionServiceImpl.java`
- `service/SysInputHistoryService.java` + `impl/SysInputHistoryServiceImpl.java`
- `service/client/PythonAlgorithmClient.java` — Python 算法服务 HTTP 客户端（重试+熔断）

**Controller**
- `controller/PredictionController.java` — 模型预测 API（3 端点）
- `controller/EvaluationController.java` — 效果评估 API（3 端点）
- `controller/ImageInputController.java` — 历史记录 API（8 端点）

**Config**
- `config/property/AlgorithmProperties.java` — Python 服务配置属性
- `config/RestClientConfig.java` — RestTemplate Bean

**Form/VO/Query**
- `model/form/AlgorithmAuditForm.java`
- `model/form/AlgorithmVersionForm.java`
- `model/form/PredictionForm.java`
- `model/form/EvaluationForm.java`
- `model/form/HistoryForm.java`
- `model/form/HistoryUpdateForm.java`
- `model/vo/AlgorithmVersionVO.java`
- `model/vo/PredictionResultVO.java`
- `model/vo/EvaluationResultVO.java`
- `model/vo/AlgorithmMonitorVO.java`
- `model/vo/PredLogVO.java`
- `model/vo/EvalLogVO.java`
- `model/vo/InputHistoryVO.java`
- `model/query/PredLogQuery.java`
- `model/query/EvalLogQuery.java`
- `model/query/HistoryQuery.java`

#### 修改文件清单（8 个）

| 文件 | 改动 |
|------|------|
| `model/entity/SysAlgorithm.java` | 新增 version/auditBy/auditTime/auditRemark 字段 |
| `service/SysAlgorithmService.java` | 新增 updateStatus/auditAlgorithm/getMonitorData/exportAlgorithmJson |
| `service/impl/SysAlgorithmServiceImpl.java` | 实现状态流转校验/审核逻辑/监控统计/SecurityContext 集成 |
| `service/SysPredLogService.java` | 新增 predict/getPredLogPage |
| `service/impl/SysPredLogServiceImpl.java` | 调用 PythonAlgorithmClient + 事务日志 |
| `service/SysEvalLogService.java` | 新增 evaluate/getEvalLogPage |
| `service/impl/SysEvalLogServiceImpl.java` | 调用 PythonAlgorithmClient + 多指标解析 |
| `controller/SysAlgorithmController.java` | 新增 11 个端点 + 权限注解 + JSON 导入校验 |
| `pom.xml` | 添加 maven-compiler-plugin annotationProcessorPaths（Lombok） |
| `application-dev.yml` | 添加 `algorithm.python.*` 配置节 |

### 2.3 Python 端新增代码

| 文件 | 作用 |
|------|------|
| `app/router/prediction.py` | `POST /api/v1/prediction` 去雾处理入口 |
| `app/router/evaluation.py` | `POST /api/v1/evaluation` 效果评估入口 |
| `app/service/prediction_service.py` | 预测编排（算法查找→图片下载→dehaze()→存储） |
| `app/router/__init__.py` | 注册 prediction + evaluation 路由 |

### 2.4 SDK 新增代码

#### 修改文件

| 文件 | 改动 |
|------|------|
| `src/api/algorithm/model.ts` | 新增 AlgorithmAuditForm/AlgorithmVersionForm/AlgorithmVersionVO/AlgorithmMonitorVO |
| `src/api/algorithm/index.ts` | 新增 11 个方法（audit/status/versions/export/import/monitor） |
| `src/api/model/model.ts` | 新增 PredictionForm/PredictionResultVO/PredLogVO/EvaluationForm/EvaluationResultVO/EvalLogVO |
| `src/api/model/index.ts` | 新增 predict/getPredTaskStatus/getPredLogs/evaluate/getEvalTaskStatus/getEvalLogs |
| `index.ts` | 新增 ImageInputHistoryAPI 导出 |

#### 新增文件

| 文件 | 作用 |
|------|------|
| `src/api/image-input/model.ts` | HistoryForm/HistoryUpdateForm/HistoryQuery/InputHistoryVO |
| `src/api/image-input/index.ts` | ImageInputHistoryAPI（8 个方法） |
| `test/factories/model.ts` | createPredictionForm/createEvaluationForm |
| `test/modules/algorithm/algorithm-new.test.ts` | 算法管理新端点测试 |
| `test/modules/model/model.test.ts` | 预测/评估 API 测试 |
| `test/modules/image-input/history.test.ts` | 历史记录 API 测试 |

---

## 三、未完成的工作

### 3.1 运行 API 集成测试 ✅ 已完成（34/34 通过）

测试已实际运行并通过：`algorithm-new.test.ts` 10/10、`model.test.ts` 9/9（2 个预测正向测试因缺测试图片/模型文件 skip）、`history.test.ts` 15/15。以下为复现步骤：

#### 步骤 1：启动 Java 后端

```bash
cd e:\DehazeSystem\dehaze-java

# .env 文件已配置好所有密码（DEHAZE_PASSWORD=12345678, JWT_SECRET_KEY=...）
# Docker 容器 MySQL/Redis/MinIO 已在运行

# 编译（已修复 Lombok 注解处理器，应零错误）
mvn compile -DskipTests

# 启动服务
mvn spring-boot:run
```

验证服务启动：
```bash
curl http://localhost:8989/api/v1/auth/captcha
# 应返回 {"code":"00000","data":{"captchaKey":"...","captchaBase64":"..."}}
```

#### 步骤 2：运行 SDK 测试

```bash
cd e:\DehazeSystem\dehaze-tool\dehaze-sdk-js

# 构建 SDK（确保 dist/ 最新）
pnpm build

# 运行所有测试
pnpm test

# 只运行新增测试
npx vitest --run test/modules/algorithm/algorithm-new.test.ts
npx vitest --run test/modules/model/model.test.ts
npx vitest --run test/modules/image-input/history.test.ts
```

#### 步骤 3：剩余可选完善

- **预测 API 正向用例**：需提供真实测试图片 + 模型文件（`.pth`）后，2 个 skip 的用例才能跑通

### 3.2 数据库建表 ✅ 已完成

`sys_algorithm_version`、`sys_input_history` 两张新表及 `sys_algorithm` 的 `version/audit_by/audit_time/audit_remark` 字段均已纳入：
- `dehaze-java/config/sql/schema.sql`（完整建表）
- `dehaze-java/config/sql/migration_20260711.sql`（增量迁移脚本）

无需再手动建表，直接执行 schema/migration 即可。

### 3.3 Python 服务联调（基本打通，仅缺测试素材）

Python 端 `app/router/prediction.py`、`evaluation.py`、`prediction_service.py` 已实现并注册，调用链已打通（loguru→logging、`async_session_factory`、`thop` 依赖等问题已修复；Java 侧 Python 服务 URL 已改为 `127.0.0.1:8014`）。剩余仅为跑通真实推理所需素材：

1. **启动 Python 服务**：`cd dehaze-python && .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8014`
2. **算法模块路径**：`prediction_service.py` 通过 `importlib.import_module(f"algorithm.{module_name}.run")` 动态导入，需确认 `import_path` 与 `algorithm/` 目录名匹配
3. **模型文件路径**：默认从 `trained_model/{module_name}/*.pth` 查找，需确认模型文件存在

### 3.4 预编译 Lombok 问题（已修复）

`pom.xml` 已添加 `maven-compiler-plugin` 的 `annotationProcessorPaths` 配置，Lombok + MapStruct 注解处理器在 Maven CLI 下正常工作。`mvn compile` 应零错误。

---

## 四、关键配置信息

### 4.1 环境变量（`dehaze-java/.env`）

```
DEHAZE_PASSWORD=12345678    # MySQL/Redis/MinIO 密码
JWT_SECRET_KEY=SecretKey012345678901234567890123456789012345678901234567890123456789
```

> **重要**：`DEHAZE_PASSWORD` 是 `12345678`（8 位），不是 `123456`。Redis/MySQL/MinIO 都用这个密码。

### 4.2 服务端口

| 服务 | 端口 |
|------|------|
| Java 后端 | 8989 |
| Python 算法服务 | 8014 |
| MySQL | 3306 |
| Redis | 6379 |
| MinIO | 9000（API）/ 9090（控制台） |

### 4.3 测试登录凭据

```
用户名：admin
密码：123456
```

### 4.4 Docker 容器状态

MySQL / Redis / MinIO 已通过 Docker 运行（`docker ps` 可见），密码均为 `12345678`。

---

## 五、新增 API 端点清单

### 5.1 算法管理新增端点（11 个）

| 方法 | 路径 | 功能 |
|------|------|------|
| PUT | `/api/v1/algorithms/{id}/status` | 修改算法状态 |
| PUT | `/api/v1/algorithms/{id}/audit` | 审核算法 |
| GET | `/api/v1/algorithms/{id}/versions` | 版本历史 |
| POST | `/api/v1/algorithms/{id}/version` | 新增版本 |
| POST | `/api/v1/algorithms/{id}/rollback` | 版本回滚 |
| GET | `/api/v1/algorithms/{id}/_export` | 导出算法 |
| POST | `/api/v1/algorithms/_export` | 批量导出 |
| POST | `/api/v1/algorithms/_import/validate` | 校验导入包 |
| POST | `/api/v1/algorithms/_import` | 导入算法 |
| GET | `/api/v1/algorithms/{id}/monitor` | 监控数据 |
| GET | `/api/v1/algorithms/{id}/monitor/stats` | 统计报表 |

### 5.2 预测 API（3 个）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/prediction` | 执行模型预测 |
| GET | `/api/v1/prediction/{taskId}` | 查询预测状态 |
| GET | `/api/v1/prediction/logs` | 预测日志列表 |

### 5.3 评估 API（3 个）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/evaluation` | 执行效果评估 |
| GET | `/api/v1/evaluation/{taskId}` | 查询评估状态 |
| GET | `/api/v1/evaluation/logs` | 评估日志列表 |

### 5.4 图像输入历史记录 API（8 个）

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/image-input/history` | 分页查询 |
| GET | `/api/v1/image-input/history/{id}` | 获取详情 |
| POST | `/api/v1/image-input/history` | 创建记录 |
| PUT | `/api/v1/image-input/history/{id}` | 更新记录 |
| DELETE | `/api/v1/image-input/history/{id}` | 删除单条 |
| DELETE | `/api/v1/image-input/history/batch` | 批量删除 |
| DELETE | `/api/v1/image-input/history/clear` | 清空全部 |
| POST | `/api/v1/image-input/history/sync` | 同步记录 |

---

## 六、测试文件说明

### 6.1 测试文件位置

```
test/modules/algorithm/algorithm-new.test.ts   — 算法管理新端点（状态/审核/版本/导出/导入/监控）
test/modules/model/model.test.ts               — 预测/评估 API（predict/evaluate/logs）
test/modules/image-input/history.test.ts       — 历史记录 CRUD + 批量删除 + 清空 + 同步
test/factories/model.ts                        — 测试数据工厂（PredictionForm/EvaluationForm）
```

### 6.2 测试模式

所有测试遵循现有项目模式：
- `beforeAll` 调用 `login()` 获取 token
- `afterAll` 调用 `logout()` 清理
- 正向测试：调用 API → 断言返回结构
- 参数校验：删除必填字段 → `expectBizErrorOrUndefined()`
- 异常测试：不存在的 ID → `expectBizErrorOrUndefined()`
- 数据清理：`afterAll` 中删除创建的测试数据

### 6.3 运行测试前提

1. Java 后端运行在 `localhost:8989`
2. MySQL/Redis/MinIO Docker 容器运行中
3. 数据库已建表（见 3.2 节 SQL）
4. `admin/123456` 用户存在且有权限

---

## 七、已知问题

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | Python 预测/评估服务未实际端到端联调 | 需真实模型文件 + `dehaze()` 输出校验 | ⏳ 待验证（Java→Python 调用链已打通） |

> 以下问题已在后续修复，保留备查：
> - `@FileExists` null NPE — 已修复（`FileExistValidator.isValid()` 对 null/空返回 true）
> - `sys_algorithm_version` / `sys_input_history` 表缺失 — 已纳入 schema.sql + migration_20260711.sql
> - `AlgorithmAPI.updateStatus` PATCH→PUT — SDK 与后端已统一为 PUT
> - `AlgorithmAPI.deleteByIds` 参数格式 — 后端 `@RequestParam List<Long> ids` 可正确绑定逗号分隔/数组，已验证

---

## 八、文件索引

### Java 后端
- 检查报告：`deliverables/dehaze-java-module-completion-report.md`
- 算法状态枚举：`src/main/java/com/pei/dehaze/common/enums/AlgorithmStatusEnum.java`
- Python 客户端：`src/main/java/com/pei/dehaze/service/client/PythonAlgorithmClient.java`
- 预测控制器：`src/main/java/com/pei/dehaze/controller/PredictionController.java`
- 评估控制器：`src/main/java/com/pei/dehaze/controller/EvaluationController.java`
- 历史记录控制器：`src/main/java/com/pei/dehaze/controller/ImageInputController.java`

### Python 端
- 预测路由：`app/router/prediction.py`
- 评估路由：`app/router/evaluation.py`
- 预测服务：`app/service/prediction_service.py`

### SDK
- 算法 API：`src/api/algorithm/index.ts`
- 模型 API：`src/api/model/index.ts`
- 历史记录 API：`src/api/image-input/index.ts`
- 算法测试：`test/modules/algorithm/algorithm-new.test.ts`
- 预测评估测试：`test/modules/model/model.test.ts`
- 历史记录测试：`test/modules/image-input/history.test.ts`

---

## 九、后续建议优先级

1. ~~P0：建表 + 运行 `pnpm test`~~ ✅ 已完成（34/34 通过，建表已入库）
2. **P1**：提供真实模型文件/测试图片 → 跑通 2 个 skip 的预测正向用例 + 端到端联调
3. **P2**：完善算法导入导出（ZIP 格式 + 模型文件打包）
4. **P2**：实现历史记录配额管理的定时清理任务
