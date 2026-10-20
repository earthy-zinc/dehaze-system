# ==================== 单测回归用测试镜像 ====================
# 构建（上下文 = 仓库根）：
#   docker build -f clarity-service/clarity-voice/Dockerfile --target test \
#     -t csighub.tencentyun.com/earthywu/clarity-vision:voice-test-<sha> .
# 依赖由独立基础镜像承载（Dockerfile.base，tag = uv.lock 指纹 3d01ce25e049），
# 本阶段只叠代码。
# 容器内布局必须保持 <root>/clarity-service/clarity-voice：app/config.py 以 pyproject.toml 向上
# 探测服务根，再取 parents[1] 作仓库根读取 /app/.env（K8s 用 Secret 挂载，见 config/kube/test/）。
# conftest 以桩替换 ASR/TTS 引擎（fake piper 模块、spy transcribe），不加载真实模型。
FROM csighub.tencentyun.com/earthywu/clarity-vision:voice-base-3d01ce25e049 AS test

WORKDIR /app/clarity-service/clarity-voice

COPY clarity-service/clarity-voice/app /app/clarity-service/clarity-voice/app
COPY clarity-service/clarity-voice/tests /app/clarity-service/clarity-voice/tests

CMD [".venv/bin/python", "-m", "pytest"]

# ==================== 运行态镜像 ====================
# clarity-voice 独立镜像：仅语音推理引擎依赖（无 MySQL/Redis/ES/MinIO 客户端，无业务代码）
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_SYSTEM_PYTHON=1 \
    # 模型落持久卷，重启不重复下载（与宿主部署的 models/ 目录语义一致）
    MODEL_CACHE_DIR=/models \
    HOST=0.0.0.0 \
    PORT=8993

# funasr/modelscope 的部分依赖无预编译 wheel 时需本地编译，保留构建工具链
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

WORKDIR /app

# 依赖层与代码层分离，代码改动不触发依赖重装。
# pyproject 的 [tool.uv.sources] 已把 torch/torchaudio 指向 PyTorch cu128 源（uv 读取该配置），
# 故基础镜像无需 NVIDIA CUDA 基座：CUDA 运行库随 wheel 的 nvidia-* 依赖装入，
# GPU 直通由宿主 nvidia-container-toolkit 提供；纯 CPU 部署可改指向 CPU 索引
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY . .

VOLUME ["/models"]
EXPOSE 8993

CMD ["python", "-m", "app.main"]
