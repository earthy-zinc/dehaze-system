# 使用NVIDIA的官方CUDA镜像作为基础镜像
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 将软件源更改为清华大学的镜像源，更新包列表并安装必要的工具
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g' /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . .

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install deploy/dependencies/torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl \
    deploy/dependencies/torchvision-0.16.0+cu121-cp310-cp310-linux_x86_64.whl \
    deploy/dependencies/natten-0.17.3+torch210cu121-cp310-cp310-linux_x86_64.whl \
    && pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple \
    && pip install -r requirements.txt \
    && mkdir -p ~/.cache/torch \
    && mkdir -p ~/.cache/huggingface \
    && mv deploy/torch/* ~/.cache/torch/ \
    && mv deploy/huggingface/* ~/.cache/huggingface/ \
    && rm -rf deploy/ \
    && rm -rf ~/.cache/pip

# 能够访问外网时，通过以下方式安装
# && pip install \
# torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
# -f https://download.pytorch.org/whl/torch_stable.html \
# && pip install \
# natten==0.15.1+torch210cu121 \
# -f https://shi-labs.com/natten/wheels/

# 创建挂载点
VOLUME /app/cache
VOLUME /app/data
VOLUME /app/trained_model

# 设置环境变量
ENV FLASK_APP=start.py
ENV FLASK_ENV=production
ENV BASICSR_JIT=True

# 暴露应用的端口
EXPOSE 80

# 启动Flask应用gunicorn -w 4 start:app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "start:app"]
