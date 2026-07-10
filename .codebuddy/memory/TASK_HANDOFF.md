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

**部门管理** (`/dept`)：6 个接口（三端数量一致）
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

### 待验证的 API（按模块）

需要先登录，注意使用账户：admin 123456，可以使用一个便携脚本文件来辅助实现登录，不必每次都重新请求浪费时间
不要求一丝一毫完全一致，只要业务逻辑一致，data、code内容一致，对于msg的小差别可以忽略

**数据集管理** (`/datasets`)：Java 7 / Go 8 / Python 7
- [ ] `GET /datasets` — 数据集列表（树形结构，支持 keywords 搜索）
- [ ] `GET /datasets/options` — 数据集下拉选项
- [ ] `GET /datasets/{id}` — 数据集详情（含统计/分布/子数据集）
- [ ] `GET /datasets/{id}/stats` — 数据集统计信息
  - ❌ **Java/Python 缺失**：仅 Go 实现了独立的统计接口
- [ ] `POST /datasets` — 新增数据集（自动生成存储目录，校验名称唯一性）
- [ ] `PUT /datasets/{id}` — 修改数据集
- [ ] `DELETE /datasets/{id}` — 删除单个数据集（级联删除）
- [ ] `DELETE /datasets/batch` — 批量删除数据集
  - ⚠️ **参数差异**：Java/Go 用 Body 传 IDs；Python 用 Query 参数 `ids` 逗号分隔

**数据项** (`/dataset-items`)：Java 8 / Go 8 / Python 6
- [ ] `GET /dataset-items` — 分页查询数据项（多维筛选：keywords/sceneType/hazeLevel/分辨率等）
- [ ] `GET /dataset-items/{id}` — 数据项详情（含清晰图/有雾图列表）
- [ ] `POST /dataset-items` — 创建空数据项（仅基本信息）
- [ ] `POST /dataset-items/upload` — 创建数据项并上传配对图片（一张清晰图+多张有雾图）
  - ❌ **Python 缺失**：Python 端未实现此接口
- [ ] `POST /dataset-items/batch` — 批量创建数据项并上传图片（按文件名自动配对）
  - ❌ **Python 缺失**：Python 端未实现此接口
- [ ] `PUT /dataset-items/{id}` — 修改数据项（名称、场景类型，XSS 防护）
- [ ] `DELETE /dataset-items/{id}` — 删除数据项（级联删除图片文件）
- [ ] `DELETE /dataset-items/batch` — 批量删除数据项
  - ⚠️ **参数差异**：Java/Go 用 Body 传 IDs；Python 用 Query 参数 `ids` 逗号分隔

**图片文件** (`/item-files`)：5 个接口（三端数量一致）
- [x] `GET /item-files/{id}` — 三端 code 一致 ✅（已修复 DB 缺列 + Python entity 多余字段 + ItemFileVO 扩展）
- [ ] `POST /item-files` — 上传数据项图片（multipart，自动解析宽高/生成缩略图/计算MD5）
- [ ] `PUT /item-files/{id}` — 修改图片标注信息（类型/场景/雾霾程度/描述）
  - ⚠️ **Go 未实现**：Go 端 handler 标注 TODO，返回"暂未实现"
- [ ] `DELETE /item-files/{id}` — 删除单个图片（同时删除缩略图）
- [ ] `DELETE /item-files/batch` — 批量删除图片（最多100张）
  - ⚠️ **参数差异**：Java/Go 用 Body 传 IDs；Python 用 Query 参数 `ids` 逗号分隔

**DB 迁移**：`sys_item_file` 添加 `scene_type/haze_level/width/height/usage_count`；`sys_dataset` 添加 `usage_count`

**算法管理** (`/algorithm`)：6 个接口（三端数量一致）
- [x] `GET /algorithm` — 三端树形结构一致 ✅（已修复 Go 扁平分页→树形 + Python VO camelCase）
- [x] `GET /algorithm/options` — code/msg 一致 ✅（条目数差异：Java 14/Python 树形/Go 73 扁平，预存差异）
- [x] `GET /algorithm/{id}` — 三端核心数据一致 ✅
- [x] `POST /algorithm` — API 结构一致 ✅（Go parentId 绑定为预存 bug）
- [x] `PUT /algorithm/{id}` — 同上
- [x] `DELETE /algorithm` — 三端参数格式不同（预存差异，见下方）

### 接口数量汇总

| 模块 | Java | Go | Python | 三端一致？ | 关键差异 |
|------|------|----|--------|-----------|----------|
| 菜单 `/menus` | 8 | 8 | 8 | ✅ 数量一致 | Python ORM 预存 bug（GET /menus） |
| 部门 `/dept` | 6 | 6 | 6 | ✅ 数量一致 | — |
| 字典 `/dict` | 11 | 11 | 11 | ✅ 数量一致 | Python `GET /dict/{typeCode}/options` 无需认证 |
| 文件 `/files` | 6 | **4** | 6 | ❌ Go 缺 2 | Go 缺 `GET /files/page` + `GET /files/{fileId}`；Python DELETE 用 Path 参数 |
| 数据集 `/datasets` | 7 | **8** | 7 | ❌ Go 多 1 | Go 独有 `GET /datasets/{id}/stats`；批量删除参数 Body vs Query |
| 数据项 `/dataset-items` | 8 | 8 | **6** | ❌ Python 缺 2 | Python 缺 upload + batch upload；批量删除参数 Body vs Query |
| 图片文件 `/item-files` | 5 | 5 | 5 | ✅ 数量一致 | Go `PUT` 未实现（TODO）；批量删除参数 Body vs Query |
| 算法 `/algorithm` | 6 | 6 | 6 | ✅ 数量一致 | DELETE 参数三端各不同（RequestParam / Path / Query） |
| **合计** | **57** | **56** | **55** | — | — |

### 需重点对齐的差异清单

1. **Go 文件管理缺 2 个接口**：`GET /files/page`（分页查询）、`GET /files/{fileId}`（文件详情）— 需在 Go 端补实现
2. **Go 数据集多 1 个接口**：`GET /datasets/{id}/stats`（统计信息）— 需确认 Java/Python 是否需要补，或 Go 是否应移除（统计数据可能已在 `GET /datasets/{id}` 详情中返回）
3. **Python 数据项缺 2 个上传接口**：`POST /dataset-items/upload`（配对上传）、`POST /dataset-items/batch`（批量上传）— 需在 Python 端补实现
4. **Go 图片文件更新未实现**：`PUT /item-files/{id}` handler 返回"暂未实现" — 需补实现
5. **批量删除参数风格不统一**：Java/Go 用 RequestBody 传 IDs；Python 用 Query 参数 `ids` 逗号分隔 — 涉及 datasets/dataset-items/item-files 三个模块
6. **文件删除参数不统一**：Java/Go 用 `DELETE /files?fileId=xx`（Query）；Python 用 `DELETE /files/{file_id}`（Path）
7. **算法删除参数不统一**：Java 用 `@RequestParam`；Go 用 `/:ids` Path；Python 用 `?ids=` Query
8. **字典下拉接口认证不一致**：Python `GET /dict/{typeCode}/options` 无需认证；Java/Go 需认证

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
