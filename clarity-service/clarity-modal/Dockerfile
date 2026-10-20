# ==================== 单测回归用测试镜像 ====================
# 构建（上下文 = 仓库根）：
#   docker build -f clarity-service/clarity-modal/Dockerfile --target test \
#     -t csighub.tencentyun.com/earthywu/clarity-vision:modal-test-<sha> .
# 依赖由独立基础镜像承载（Dockerfile.base，tag = uv.lock 指纹 08dc5a158a40），
# 本阶段只叠代码与 algorithm/。
# 容器内布局必须保持 <root>/clarity-service/clarity-modal：app/config.py 以 pyproject.toml 向上
# 探测服务根，再取 parents[1] 作仓库根读取 /app/.env（K8s 用 Secret 挂载，见 config/kube/test/）。
# 测试真跑 algorithm.DCP（无权重、纯 numpy/opencv，CPU 即可），故必须带上 algorithm/ 目录；
# 带权重的算法在测试中不真实下载与执行（conftest 把 MODEL_CACHE_DIR 指向 tmp_path、
# 权重源指向 127.0.0.1，断言走 503 分支）。
FROM csighub.tencentyun.com/earthywu/clarity-vision:modal-base-08dc5a158a40 AS test

WORKDIR /app/clarity-service/clarity-modal

COPY clarity-service/clarity-modal/algorithm /app/clarity-service/clarity-modal/algorithm
COPY clarity-service/clarity-modal/app /app/clarity-service/clarity-modal/app
COPY clarity-service/clarity-modal/tests /app/clarity-service/clarity-modal/tests

CMD [".venv/bin/python", "-m", "pytest"]

# ==================== 运行态镜像 ====================
# clarity-modal 独立镜像：仅推理依赖（无 MySQL/Redis/ES/MinIO 客户端，无业务代码）。
# 基座与 clarity-python 的算法段对齐：CUDA + cuDNN devel（RIDCP/WPXNet/CFENViTDehazing 的
# DCN 算子在首次使用时 JIT 编译，需要 nvcc 与 ninja），natten 也要求 cuDNN 运行时。
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_PYTHON=3.11 \
    # 权重与 JIT 编译缓存落持久卷，重启不重复下载/编译
    MODEL_CACHE_DIR=/models \
    TORCH_EXTENSIONS_DIR=/models/.torch_extensions \
    HOST=0.0.0.0 \
    PORT=8994

# libgl1/libglib2.0-0：opencv（DCP/MADN 等算法）运行时依赖
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g' /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（快速 Python 包管理器）；解释器由 uv 按 .python-version 自动获取
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

WORKDIR /app

# 依赖层与代码层分离，代码改动不触发依赖重装
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --no-dev --no-cache

COPY . .

VOLUME ["/models"]
EXPOSE 8994

CMD [".venv/bin/python", "-m", "app.main"]
