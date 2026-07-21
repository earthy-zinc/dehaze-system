# scripts

DehazeSystem 通用辅助脚本集合。

## 脚本说明

### run.py（服务生命周期管理，docker-like）

后端服务统一启动/停止/重启脚本，跨平台（Windows / macOS / Linux）。Java/Go/Python 三端均已纳入管理，Go 启动前自动 `go build`，Java 走 `mvn spring-boot:run`，Python 走 `.venv` 下的 uvicorn。

**直接调用：**

```bash
python scripts/run.py <command> [args...]
```

**或通过薄壳调用（体验等同三套独立脚本）：**

| 平台 | 调用方式 |
| --- | --- |
| Windows cmd / PowerShell | `scripts\run.cmd run dehaze-go` |
| Bash / Git Bash / WSL | `./scripts/run run dehaze-go` |

**命令：**

| 命令 | 用途 |
| --- | --- |
| `run <svc>[,svc...]` | 启动服务（Go 自动编译） |
| `stop <svc>[,svc...]` | 停止服务（先按 PID 文件，再按端口 fallback） |
| `restart <svc>[,svc...]` | 重启服务 |
| `ps` | 查看所有服务状态（端口 / PID / 日志路径） |
| `logs <svc> [lines]` | 查看服务日志（默认 50 行） |
| `kill <port>` | 杀掉占用指定端口的进程 |

**服务名（支持别名 / 逗号分隔 / `all`）：**

| 服务 | 别名 | 端口 |
| --- | --- | --- |
| `dehaze-go` | `go` | 8990 |
| `dehaze-python` | `python` | 8991 |
| `dehaze-java` | `java` | 8989 |

**示例：**

```bash
python scripts/run.py run dehaze-go
python scripts/run.py run dehaze-go,dehaze-java
python scripts/run.py stop all
python scripts/run.py restart dehaze-go
python scripts/run.py ps
python scripts/run.py logs dehaze-python 100
python scripts/run.py kill 8990
```

> PID 文件分别落在各服务目录（`dehaze-go/go_server.pid` 等），日志同理。

### login_helper.py

三端登录辅助脚本，自动完成「获取验证码 → 从 Redis 读取验证码 → 登录」流程，输出 `accessToken`，便于后续 curl/脚本调用。

```bash
python scripts/login_helper.py [go|python|java|all]
# 默认 go；all 会打印三端 token 及一段 JSON
```

- 账号：`admin / 123456`
- Java captcha 存 Redis db0（Jackson 序列化带引号，已自动去除）
- Go/Python captcha 存 Redis db3

### debug_helper.py

三端 API 调试辅助脚本，封装调试中常用的重复操作。内部依赖 `login_helper.py` 自动获取 token。

```bash
python scripts/debug_helper.py <command> [args...]
```

| 命令 | 用途 |
| --- | --- |
| `status` | 查看三端服务（Java 8989 / Go 8990 / Python 8991）运行状态 |
| `restart <go\|python\|all>` | 重启后端服务（Go 会自动编译） |
| `build <go>` | 编译指定后端 |
| `compare <path> [method] [body]` | 三端对比同一 API，并输出一致性判断 |
| `curl <backend> <path> [method] [body]` | 单端请求 |
| `db <sql>` | 在 MySQL（容器 `mysql`，库 `dehaze`）执行 SQL |
| `redis <get\|keys> <key> [db]` | Redis（容器 `redis`）操作 |
| `logs <python\|go>` | 查看服务日志（最后 30 行） |
| `kill <port>` | 杀掉占用指定端口的进程 |

示例：

```bash
python scripts/debug_helper.py status
python scripts/debug_helper.py restart go
python scripts/debug_helper.py compare /api/v1/users/page
python scripts/debug_helper.py curl go /api/v1/users/page
python scripts/debug_helper.py db "SELECT id,username FROM sys_user WHERE deleted=0"
```

> Git Bash 中执行 `compare` 需加 `MSYS_NO_PATHCONV=1` 前缀避免路径转换。

### refresh_dataset_db.py

数据集数据库刷新脚本：扫描 `D:\DeepLearning\dataset` 下的叶子数据集，重建 `sys_file` / `sys_dataset_item` / `sys_item_file` 三张表。

- 文件 URL 直连 nginx-dataset：`http://127.0.0.1:9000/{object_name}`
- 配对策略：按文件名前导数字分组（如 `01_GT.png` 与 `01_hazy.png` 归到 `"01"` 组）
- MD5 字段使用 `object_name` 的 MD5（保证 UNIQUE 约束，不读文件内容以提速）
- 前置依赖：`.env` 中已设置 `DEHAZE_PASSWORD`，docker-compose 已启动 MySQL + nginx-dataset

```bash
python scripts/refresh_dataset_db.py
```
