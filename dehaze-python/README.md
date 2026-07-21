# 图像去雾系统 (Python 算法服务)

基于 PyTorch 构建深度学习模型、FastAPI 异步 Web 框架、Uvicorn 部署的图像去雾算法服务，提供 API 接口供 Java/Go 后端调用。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Python算法服务架构文档.md)。

## 技术栈

- **Web 框架**: FastAPI
- **ASGI 服务器**: Uvicorn
- **深度学习**: PyTorch + CUDA 12.1
- **依赖管理**: uv + pyproject.toml
- **容器化**: Docker（多阶段构建）
- **对象存储**: MinIO
- **监控**: Prometheus + Grafana
- **实时通信**: WebSocket

## 快速开始

### 一键启动（推荐）

项目提供了 `start.sh` 脚本，自动完成虚拟环境激活、依赖同步、旧进程清理和后台启动：

```bash
./start.sh
```

脚本支持通过环境变量自定义参数：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEHAZE_PYTHON_PORT` | `8991` | 监听端口 |
| `DEHAZE_PYTHON_HOST` | `0.0.0.0` | 监听地址 |
| `DEHAZE_PYTHON_WORKERS` | `1` | uvicorn worker 进程数 |

示例：

```bash
DEHAZE_PYTHON_PORT=9000 DEHAZE_PYTHON_WORKERS=4 ./start.sh
```

启动后日志输出到 `logs/dehaze-python.log`，PID 文件位于 `logs/dehaze-python.pid`。

### 手动启动

```bash
# 创建虚拟环境并安装依赖
uv venv .venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate
uv sync

# 开发模式（热重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8991

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8991 --workers 4
```

## 访问地址

- API 文档: `http://localhost:8991/docs`
- ReDoc: `http://localhost:8991/redoc`
- 健康检查: `http://localhost:8991/health`
