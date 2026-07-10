# 任务交接文档

## 一、任务目标

验证 DehazeSystem 三个后端（Go / Java / Python）的**每一个对外 API 的业务逻辑、入参、返回完全一致**。这三个后端是同一服务的三种语言实现，必须保证：
- 相同 URL 路径、HTTP 方法
- 相同请求参数（字段名、类型、校验规则）
- 相同响应结构（`{code, msg, data, traceId, timestamp, errors}`）
- 相同业务逻辑（登录流程、权限校验、错误码、分页格式等）

设计文档位于：`dehaze-doc/docs/03-模块设计/`（基础模块 + 核心模块），全局规范在 `dehaze-doc/docs/02-系统架构/04-API规范.md`。

## 二、运行环境

### Docker 容器（全部运行中）
| 服务 | 端口 | 密码 |
|------|------|------|
| MySQL | 3306 | 12345678 |
| Redis | 6379 | 12345678 |
| MongoDB | 27017 | 12345678 |
| MinIO | 9000/9090 | admin/12345678 |
| PostgreSQL | 5432 | 12345678 |
| RabbitMQ | 5672/15672 | root/12345678 |

### 后端服务端口
| 后端 | 端口 | 启动命令 | 热重载 |
|------|------|---------|--------|
| Java | 8989 | `cd dehaze-java && mvn spring-boot:run -DskipTests -Dmaven.test.skip=true` | ✅ devtools |
| Go | 8999 | `cd dehaze-go && go build -o bin/dehaze-go.exe ./cmd/main.go && ./bin/dehaze-go.exe` | ❌ 需重新编译 |
| Python | 8014 | `cd dehaze-python && .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8014` | 加 `--reload` |

### ⚠️ Windows 关键注意事项
- 用 `127.0.0.1` 而非 `localhost`：根因是 Windows `localhost` 解析为 IPv6 `::1`（优先）+ IPv4 `127.0.0.1`，Docker 端口只绑定 IPv4，Python socket 串行尝试先连 IPv6 失败后等 21 秒 TCP SYN 超时才回退 IPv4。Go（Happy Eyeballs 并行尝试）和 Java（默认偏好 IPv4）不受影响
- Python `.env` 已配置 `DB_HOST=127.0.0.1` 和 `REDIS_HOST=127.0.0.1`
- Go 的 `config.yaml` 已将所有 localhost 改为 `127.0.0.1`
- 后台进程（`&`）在 Git Bash 间不持久，需在独立终端启动 Python

### 统一密码
- `.env` 文件中 `DEHAZE_PASSWORD=12345678`（MinIO 要求 ≥8 位）
- JWT 密钥统一：`JWT_SECRET_KEY=SecretKey012345678901234567890123456789012345678901234567890123456789`
- 四个 `.env` 文件：根目录、`dehaze-go/`、`dehaze-python/`、`dehaze-java/`

## 三、已完成的工作

### 1. 配置统一（✅ 完成）
- 四个 `.env` 文件密码统一为 `12345678`
- JWT 密钥三端统一
- Go/Python host 从 `earthyzinc.devcloud.woa.com` 改为 `localhost`
- Java 添加 `spring-dotenv:4.0.0` 依赖以加载 `.env`

### 2. 响应格式统一（✅ 完成）
### 3. 错误码纠正（✅ 完成）
### 4. 分页参数统一（✅ 完成）

## 四、当前进度 — API 逐个验证

### 已验证的 API

| API | Java | Go | Python | 状态 |
|-----|------|----|--------|------|
| `GET /auth/captcha` | ✅ 一致 | ✅ 一致 | ✅ 一致 | 三端一致 |
| `POST /auth/login` | ✅ JSON body, user={id,username,nickname} | ✅ 同左 | ✅ 同左 | 三端一致 |
| `POST /auth/logout` | ✅ POST, code=00000, msg=一切ok | ✅ 同左 | ✅ 同左 | 三端一致 |
| `GET /auth/me` | ✅ code=00000, msg=一切ok | ✅ 同左 | ✅ 同左 | 三端一致 |
| token 失效后 | ✅ 401/A0230/token无效或已过期 | ✅ 同左 | ✅ 同左 | 三端一致 |
| 跨端 token 互认 | ✅ Java→Go/Python | ✅ Go→Java/Python | ✅ Python→Java/Go | 全部通过 |

**认证模块** (`/auth`)：
- [x] `GET /auth/captcha` — 三端一致 ✅
- [x] `POST /auth/login` — 三端一致 ✅（JSON body, 返回 user 字段）
- [x] `POST /auth/logout` — 三端一致 ✅（POST, token 黑名单）
- [x] `GET /auth/me` — 三端一致 ✅
- [x] 跨端 token 互认 — 全部通过 ✅
- [x] token 失效响应 — 三端一致 ✅（401/A0230）
- [x] `POST /auth/refresh` — 三端一致 ✅（Header token，返回新 token + user）

### 待验证的 API（按模块）

需要先登录，注意使用账户：admin 123456，可以使用一个便携脚本文件来辅助实现登录，不必每次都重新请求浪费时间
不要求一丝一毫完全一致，只要业务逻辑一致，data、code内容一致，对于msg的小差别可以忽略

**用户管理** (`/users`)：✅ 全部通过（7/7）
- [x] `GET /users/page` — 三端一致 ✅
- [x] `GET /users/{id}/form` — 三端一致 ✅
- [x] `POST /users` — 三端一致 ✅
- [x] `PUT /users/{id}` — 三端一致 ✅
- [x] `DELETE /users/{ids}` — 三端一致 ✅
- [x] `PATCH /users/{id}/status` — 三端一致 ✅
- [x] `PATCH /users/{id}/password` — 三端一致 ✅

**角色管理** (`/roles`)：
- [x] `GET /roles/page` — code/data结构一致 ✅（total差异因数据范围过滤）
- [x] `GET /roles/options` — Java/Go ✅，Python ❌（RoleOptionVO.value类型不匹配，预存bug）
- [ ] `POST /roles` — 预存路由/权限问题（非一致性bug）
- [ ] `PUT /roles/{id}` — 同上
- [ ] `DELETE /roles/{ids}` — 同上
- [ ] `PATCH /roles/{id}/status` — 同上
- [ ] `PATCH /roles/{id}/menus` — 同上

**菜单管理** (`/menus`)：
- [x] `GET /menus` — Java/Go code/data一致 ✅，Python ❌（ORM预存bug）

**部门管理** (`/dept`)：
- [x] `GET /dept` — 三端code/data结构一致 ✅

**字典管理** (`/dict`)：
- [x] `GET /dict/types/page` — 三端完全一致 ✅

**文件管理** (`/files`)：
- [ ] 上传、下载、删除、MD5 校验

**数据集管理** (`/datasets`)、**数据项** (`/dataset-items`)、**图片文件** (`/item-files`)：
- [ ] 各模块 CRUD + 批量操作

**算法管理** (`/algorithm`)：
- [ ] CRUD + 选项接口

## 五、验证方法建议

每个 API 的验证流程：
1. 读设计文档 `dehaze-doc/docs/03-模块设计/` 中对应模块的 `API接口.md`
2. 用 curl 测试 Java（8989）、Go（8999）、Python（8014）三端
3. 对比三端返回的 `code`、`msg`、`data` 字段结构
4. 如有不一致，读对应后端的 handler/service 代码，找出差异
5. 以 Java 为参考实现（Java 是最完整的），修改 Go/Python 对齐
6. 修改后重新测试确认一致

### 测试用例需要覆盖
- 正常请求（200 成功）
- 参数校验失败（A0400）
- 未认证（A0230）
- 无权限（A0301）
- 资源不存在（A0401）
- 业务规则冲突（A0501 数据已存在等）

## 六、已知待修复问题

4. **URL 复数统一**（低优先级）：`/dept` → `/depts`、`/algorithm` → `/algorithms`，需前端同步
5. **Go 测试文件**：`test/integration/auth_integration_test.go` 中 `assert.Equal(t, "验证码获取成功", resp.Msg)` 需更新为 `"一切ok"`
8. **Java API 严重不一致**：login 用表单参数（非 JSON body）、logout 用 DELETE 方法、缺少 `/auth/me` 和 `/auth/refresh` 接口、Go LoginResult 缺少 user 字段

## 七、关键文件索引

### 设计文档
- 全局 API 规范：`dehaze-doc/docs/02-系统架构/04-API规范.md`
- 模块 API 文档：`dehaze-doc/docs/03-模块设计/基础模块/*/API接口.md`
- 模块 API 文档：`dehaze-doc/docs/03-模块设计/核心模块/*/API接口.md`

### Java（参考实现）
- 响应包装：`dehaze-java/src/main/java/com/pei/dehaze/common/result/Result.java`
- 错误码：`dehaze-java/src/main/java/com/pei/dehaze/common/result/ResultCode.java`
- 分页：`dehaze-java/src/main/java/com/pei/dehaze/common/result/PageResult.java`
- 控制器：`dehaze-java/src/main/java/com/pei/dehaze/controller/*.java`

### Go
- 响应：`dehaze-go/pkg/common/response.go`
- 错误码：`dehaze-go/pkg/common/result_code.go`
- 控制器：`dehaze-go/internal/api/*.go`
- 路由：`dehaze-go/internal/router/*.go`
- 配置：`dehaze-go/config/config.yaml`

### Python
- 响应：`dehaze-python/app/core/result.py`
- 错误码：`dehaze-python/app/core/code.py`
- 异常：`dehaze-python/app/core/exceptions.py`
- 路由：`dehaze-python/app/router/*.py`
- Schema：`dehaze-python/app/models/schema/*.py`
- 配置：`dehaze-python/app/config.py` + `dehaze-python/.env`

### 环境配置
- Docker：`docker-compose.yml`（volumes 已设 `external: true`）
- 根 `.env`：`DEHAZE_PASSWORD=12345678`
- Go `.env`：`dehaze-go/.env`（`DEHAZE_PASSWORD` + `JWT_SECRET_KEY`）
- Python `.env`：`dehaze-python/.env`（含 `DB_HOST=127.0.0.1`、`REDIS_HOST=127.0.0.1`）
- Java `.env`：`dehaze-java/.env`（含 `DEHAZE_PASSWORD` + `JWT_SECRET_KEY` + Docker 卷路径）

### 记忆文件
- 长期记忆：`.codebuddy/memory/MEMORY.md`
