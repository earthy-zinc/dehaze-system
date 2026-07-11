# 任务交接文档

## 一、任务背景

DehazeSystem 有三个后端（Go / Java / Python），是同一服务的三种语言实现，需保证对外 API 在 URL 路径、HTTP 方法、请求参数、响应结构（`{code, msg, data, traceId, timestamp, errors}`）、业务逻辑上一致。

> 注意：三端存在**预存差异**（如 Go 算法选项 73 条扁平 vs Java/Python 树形、各端个别冗余字段），这些是历史设计差异，**不应盲目"对齐修复"**，改动前先确认是否为有意为之。

设计文档：`dehaze-doc/docs/03-模块设计/`，全局规范：`dehaze-doc/docs/02-系统架构/04-API规范.md`。

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

## 三、API 一致性验证方法

每次需要复核三端一致性时：
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

## 四、关键文件索引

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
