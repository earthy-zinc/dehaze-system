# ==================== 单测回归用测试镜像 ====================
# 构建（上下文 = 仓库根）：
#   docker build -f clarity-python/Dockerfile --target test \
#     -t csighub.tencentyun.com/earthywu/clarity-vision:py-test-<sha> .
#
# 为什么不再保留独立依赖基础镜像（原 Dockerfile.base / py-base-v2 已删除）：
#   推理能力外置为 clarity-service/* 后，编排侧 pyproject 已零 GPU 依赖（torch / nvidia / natten /
#   llama-cpp / funasr / piper 全移除，opencv 锁 4.x），lock 从 529 包降到 160 包、依赖体积从
#   9.3G 降到 GB 级 ⇒ 独立 base 镜像省下的重复构建时间抵不过多维护一个 tag，依赖层放在本文件内
#   靠 Docker 层缓存复用（pyproject/uv.lock 不变即命中）即可。
# 与运行态镜像（下方）的差异：上下文用仓库根（conftest 从仓库根 config/sql 重建测试库）；
# 不含 .env（凭证由 K8s Secret 挂载到 /app/.env，见 config/kube/test/job-regression.yaml）。
FROM python:3.11-slim AS test

ENV LANG=C.UTF-8 \
    TZ=Asia/Shanghai \
    UV_LINK_MODE=copy

# libgomp1 / libgl1 / libglib2.0-0t64：opencv（rapidocr 传递引入，锁 4.x）的运行库
RUN sed -i 's@//deb.debian.org@//mirrors.cloud.tencent.com@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl git libgomp1 libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# 容器内布局与宿主检出路径无关：/app/clarity-python + /app/config + /app/.env（Secret 挂载）
WORKDIR /app/clarity-python

# 依赖层：uv.lock 不变即命中缓存。必须 --no-cache，否则 uv 把已下载轮子留在 /root/.cache/uv
# 永久占住镜像层（此前实测 10.2G → 19.7G）。
# --extra test 取 [project.optional-dependencies].test；默认同时装 [dependency-groups].dev（fakeredis/ruff）。
COPY clarity-python/pyproject.toml clarity-python/uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project --extra test

# 代码层：GPU 推理代码（algorithm/）已迁至 clarity-service/clarity-modal，此处不再拷贝
COPY clarity-python/app /app/clarity-python/app
COPY clarity-python/tests /app/clarity-python/tests
COPY clarity-python/scripts /app/clarity-python/scripts
COPY clarity-python/pytest.ini /app/clarity-python/pytest.ini
COPY config /app/config

# 依赖自检：坏环境在构建期暴露，不留到 Pod 才失败
RUN .venv/bin/python -c "import fastapi, sqlalchemy, aiomysql, motor, elasticsearch, minio, cv2, rapidocr_onnxruntime, fakeredis, langgraph; print('[deps-ok]')" \
    && .venv/bin/python -m pytest --version

CMD [".venv/bin/python", "-m", "pytest"]

# ==================== 运行态镜像 ====================
# 使用 NVIDIA CUDA 镜像作为基础镜像
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖：libgl1/libglib2.0-0 为 rapidocr 传递引入的 opencv 运行库；curl 用于容器健康检查
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（快速 Python 包管理器）
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

# 设置环境变量
ENV UV_SYSTEM_PYTHON=1
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV APP_ENV=production
# 容器内服务端口（覆盖 config.py 默认的 8991）
ENV SERVER_PORT=80
# Prometheus 多进程指标聚合（配合多 Worker 使用）
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

# 设置工作目录
WORKDIR /app

# 复制依赖文件（利用 Docker 缓存层）
COPY pyproject.toml .

# 安装 Python 依赖（GPU 推理依赖已随 clarity-service/* 迁出，无需 CUDA 源与 torch 预装）
RUN uv pip install --system -e .

# 复制项目文件
COPY . .

# 数据集本地存储挂载点
VOLUME /app/dataset

# 暴露端口
EXPOSE 80

# 启动 FastAPI 应用（参数由 app/config.py + 环境变量驱动）
CMD ["python", "-m", "app.main"]
