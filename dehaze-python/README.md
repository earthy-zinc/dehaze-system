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
