# 使用 NVIDIA CUDA 镜像作为基础镜像
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖和 Python 3.10
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g' /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 安装 uv（快速 Python 包管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 设置环境变量
ENV UV_SYSTEM_PYTHON=1
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV APP_ENV=production
ENV BASICSR_JIT=True

# 设置工作目录
WORKDIR /app

# 复制依赖文件（利用 Docker 缓存层）
COPY pyproject.toml .

# 安装 Python 依赖
RUN uv pip install --system torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --system -e . \
    && mkdir -p ~/.cache/torch \
    && mkdir -p ~/.cache/huggingface

# 复制项目文件
COPY . .

# 创建挂载点
VOLUME /app/trained_model
VOLUME /app/dataset

# 暴露端口
EXPOSE 80

# 启动 FastAPI 应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
