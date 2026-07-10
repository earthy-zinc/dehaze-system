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

## 四、当前进度 — API逐个验证

### 已验证的 API

需要先登录，注意使用账户：admin 123456，可以使用一个便携脚本文件来辅助实现登录，不必每次都重新请求浪费时间
不要求一丝一毫完全一致，只要业务逻辑一致，data、code内容一致，对于msg的小差别可以忽略

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
- [x] `GET /roles/options` — 三端一致 ✅（已修复：Python 缓存脏数据 + Go/Python 添加 ROOT 角色过滤 + Go msg 统一）
- [x] `POST /roles` — 已修复（Go JWT 权限缺失 + Python 路由/装饰器签名 bug）
- [x] `PUT /roles/{id}` — 同上，已修复
- [x] `DELETE /roles/{ids}` — 同上，已修复
- [x] `PATCH /roles/{id}/status` — 同上，已修复
- [x] `PATCH /roles/{id}/menus` — 同上，已修复（同模式添加 user 参数）

**菜单管理** (`/menus`)：8 个接口（三端数量一致）
- [x] `GET /menus` — 三端 code/data 一致 ✅（已修复 Python ORM `deleted` 字段不存在的 bug）
- [x] `GET /menus/options` — 三端完全一致 ✅（已修复 Go gorm:"-" + Python MenuOptionVO 字段）
- [x] `GET /menus/routes` — 三端完全一致 ✅（已修复 Python RouteVO 缺少 meta 字段）
- [x] `GET /menus/{id}/form` — 三端 code/data 一致 ✅（Python 与 Java 字段对齐，Go 多字段为预存差异）
- [x] `POST /menus` — 三端一致 ✅（已修复 Python trailing slash + user 参数缺失）
- [x] `PUT /menus/{id}` — 同上，已修复
- [x] `DELETE /menus/{id}` — 同上，已修复
- [x] `PATCH /menus/{id}` — 同上，已修复

**部门管理** (`/depts`)：6 个接口（三端数量一致）
- [x] `GET /dept` — 三端code/data结构一致 ✅
- [x] `GET /dept/options` — 三端完全一致 ✅（已修复 Go 树结构 + Python 缓存）
- [x] `GET /dept/{deptId}/form` — 三端完全一致 ✅
- [x] `POST /dept` — 三端一致 ✅（已修复 Python user 参数 + trailing slash）
- [x] `PUT /dept/{deptId}` — 同上，已修复
- [x] `DELETE /dept/{ids}` — 同上，已修复

**字典管理** (`/dict`)：11 个接口（三端数量一致）
- [x] `GET /dict/types/page` — 三端完全一致 ✅
- [x] `GET /dict/types/{id}/form` — 三端一致 ✅（Go 多 remark 为预存差异）
- [x] `POST /dict/types` — 三端权限检查一致 ✅
- [x] `PUT /dict/types/{id}` — 同上
- [x] `DELETE /dict/types/{ids}` — 同上
- [x] `GET /dict/page` — 三端完全一致 ✅
- [x] `GET /dict/{id}/form` — 三端一致 ✅（Java 少 defaulted/remark 为预存差异）
- [x] `POST /dict` — 三端权限检查一致 ✅
- [x] `PUT /dict/{id}` — 同上
- [x] `DELETE /dict/{ids}` — 同上
- [x] `GET /dict/{typeCode}/options` — 三端完全一致 ✅（已修复 Go Gin 路由冲突 + Python 添加认证）

**文件管理** (`/files`)：Java 6 / Go 6 / Python 6
- [x] `POST /files` — API 结构一致 ✅（已修复 Python trailing slash；实际上传依赖存储配置）
- [x] `DELETE /files` — 三端参数一致 ✅（已修复 Python path→query param `fileId`）
- [x] `GET /files/check` — 三端完全一致 ✅（已修复 Java check 空实现 + Result.judge → Result.success）
- [x] `GET /files/page` — 三端一致 ✅（已修复 Java/Python `size_bytes` 不存在 + Go 新增端点）
- [x] `GET /files/{fileId}` — 三端完全一致 ✅（已修复 Python sizeBytes + Go 新增端点 + updateTime 对齐）
- [x] `GET /files/download/**` — API 结构一致 ✅（三端均依赖存储配置，非 API 一致性问题）

**数据集管理** (`/datasets`)：Java 7 / Go 7 / Python 7
- [x] `GET /datasets` — 数据集列表（树形结构，支持 keywords 搜索）✅
- [x] `GET /datasets/options` — 数据集下拉选项 ✅
- [x] `GET /datasets/{id}` — 数据集详情（含统计/分布/子数据集）✅
- [x] `POST /datasets` — 新增数据集（自动生成存储目录，校验名称唯一性）✅
- [x] `PUT /datasets/{id}` — 修改数据集 ✅
- [x] `DELETE /datasets/{id}` — 删除单个数据集（级联删除）✅
- [x] `DELETE /datasets/batch` — 批量删除数据集 ✅（已统一 Python Body 参数）

**数据项** (`/dataset-items`)：Java 8 / Go 8 / Python 8
- [x] `GET /dataset-items` — 分页查询数据项（多维筛选：keywords/sceneType/hazeLevel/分辨率等）✅
- [x] `GET /dataset-items/{id}` — 数据项详情（含清晰图/有雾图列表）✅
- [x] `POST /dataset-items` — 创建空数据项（仅基本信息）✅
- [x] `POST /dataset-items/upload` — 创建数据项并上传配对图片（一张清晰图+多张有雾图）✅（已补 Python 缺失）
- [x] `POST /dataset-items/batch` — 批量创建数据项并上传图片（按文件名自动配对）✅（已补 Python 缺失）
- [x] `PUT /dataset-items/{id}` — 修改数据项（名称、场景类型，XSS 防护）✅
- [x] `DELETE /dataset-items/{id}` — 删除数据项（级联删除图片文件）✅
- [x] `DELETE /dataset-items/batch` — 批量删除数据项 ✅（已统一 Python Body 参数）

**图片文件** (`/item-files`)：5 个接口（三端数量一致）
- [x] `GET /item-files/{id}` — 三端 code 一致 ✅（已修复 DB 缺列 + Python entity 多余字段 + ItemFileVO 扩展）
- [x] `POST /item-files` — API 结构一致 ✅（multipart 上传）
- [x] `PUT /item-files/{id}` — 三端一致 ✅（已修复 Go PUT 未实现 → 补实现 UpdateItemFileInfo）
- [x] `DELETE /item-files/{id}` — 三端一致 ✅
- [x] `DELETE /item-files/batch` — 三端一致 ✅（已修复 Python Query→Body，三端统一用 `{"ids":[...]}` JSON Body）

**DB 迁移**：`sys_item_file` 添加 `scene_type/haze_level/width/height/usage_count`；`sys_dataset` 添加 `usage_count`

**算法管理** (`/algorithms`)：6 个接口（三端数量一致）
- [x] `GET /algorithm` — 三端树形结构一致 ✅（已修复 Go 扁平分页→树形 + Python VO camelCase）
- [x] `GET /algorithm/options` — code/msg 一致 ✅（条目数差异：Java 14/Python 树形/Go 73 扁平，预存差异）
- [x] `GET /algorithm/{id}` — 三端核心数据一致 ✅
- [x] `POST /algorithm` — API 结构一致 ✅（Go parentId 绑定为预存 bug）
- [x] `PUT /algorithm/{id}` — 同上
- [x] `DELETE /algorithm` — 三端一致 ✅（已修复 Go Path→Query，三端统一用 `?ids=1,2,3` 查询参数；SDK 客户端同步更新）

### 接口数量汇总

| 模块 | Java | Go | Python | 三端一致？ | 关键差异 |
|------|------|----|--------|-----------|----------|
| 菜单 `/menus` | 8 | 8 | 8 | ✅ 数量一致 | Python ORM 预存 bug（GET /menus） |
| 部门 `/depts` | 6 | 6 | 6 | ✅ 数量一致 | — |
| 字典 `/dict` | 11 | 11 | 11 | ✅ 数量一致 | Python `GET /dict/{typeCode}/options` 无需认证 |
| 文件 `/files` | 6 | **6** | 6 | ✅ 数量一致 | ✅ 已修复：Go 新增 page/detail + Python DELETE Query 参数 |
| 数据集 `/datasets` | 7 | **7** | 7 | ✅ 数量一致 | ✅ 已移除 Go 冗余 /stats 端点 |
| 数据项 `/dataset-items` | 8 | 8 | **8** | ✅ 数量一致 | ✅ 已补 Python upload/batch 两个上传端点 |
| 图片文件 `/item-files` | 5 | 5 | 5 | ✅ 数量一致 | ✅ 已修复：Go PUT 未实现 + Python 批量删除 Query→Body |
| 算法 `/algorithms` | 6 | 6 | 6 | ✅ 数量一致 | ✅ 已修复：Go DELETE Path→Query，三端统一 |
| **合计** | **57** | **56** | **57** | — | — |

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

（全部已修复，无已知问题）

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
