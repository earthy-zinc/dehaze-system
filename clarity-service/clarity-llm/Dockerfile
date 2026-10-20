# ==================== 单测回归用测试镜像 ====================
# 构建（上下文 = 仓库根）：
#   docker build -f clarity-service/clarity-llm/Dockerfile --target test \
#     -t csighub.tencentyun.com/earthywu/clarity-vision:llm-test-<sha> .
#
# 为什么不单独做依赖基础镜像：lock 仅 41 包（llama-cpp-python 为 sdist，构建期本地编译 CPU 版），
# 独立 base 省下的重复构建时间抵不过多维护一个 tag，依赖层靠 Docker 层缓存复用即可。
# 容器内布局必须保持 <root>/clarity-service/clarity-llm：app/config.py 以 pyproject.toml 向上探测
# 服务根，再取 parents[1] 作仓库根读取 /app/.env（K8s 用 Secret 挂载，见 config/kube/test/）。
FROM python:3.11-slim AS test

ENV LANG=C.UTF-8 \
    TZ=Asia/Shanghai \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# build-essential/cmake：llama-cpp-python 无预编译 wheel 时本地编译
RUN sed -i 's@//deb.debian.org@//mirrors.cloud.tencent.com@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

WORKDIR /app/clarity-service/clarity-llm

# 依赖层：uv.lock 不变即命中缓存。--no-cache 防止 uv 下载缓存永久占住镜像层。
COPY clarity-service/clarity-llm/pyproject.toml clarity-service/clarity-llm/uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# 代码层：conftest 以桩替换 llama_cpp，不加载真实 GGUF 权重，无需 models/
COPY clarity-service/clarity-llm/app /app/clarity-service/clarity-llm/app
COPY clarity-service/clarity-llm/tests /app/clarity-service/clarity-llm/tests

# 依赖自检：坏环境在构建期暴露，不留到 Pod 才失败
RUN .venv/bin/python -c "import fastapi, llama_cpp, respx, pytest_asyncio; print('[deps-ok]')" \
    && .venv/bin/python -m pytest --version

CMD [".venv/bin/python", "-m", "pytest"]

# ==================== 运行态镜像 ====================
# clarity-llm 独立镜像：仅推理依赖（无 MySQL/Redis/ES/MinIO 客户端，无业务代码）
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_SYSTEM_PYTHON=1 \
    # 模型落持久卷，重启不重复下载（与宿主部署的 models/ 目录语义一致）
    MODEL_CACHE_DIR=/models \
    HOST=0.0.0.0 \
    PORT=8992

# llama-cpp-python 无预编译 wheel 时需本地编译，保留构建工具链
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

WORKDIR /app

# 依赖层与代码层分离，代码改动不触发依赖重装
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY . .

VOLUME ["/models"]
EXPOSE 8992

CMD ["python", "-m", "app.main"]
