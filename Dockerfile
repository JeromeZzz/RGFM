# --------------------------------------------------------------------------------
# 1. 基础镜像：CUDA 12.6 + Ubuntu 24.04 (适配 H200)
# --------------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# --------------------------------------------------------------------------------
# 2. 环境变量设置
# --------------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# H200 (Hopper) 需要 Compute Capability 9.0
ENV TORCH_CUDA_ARCH_LIST="9.0"
# 避免 apt 交互
ENV DEBIAN_FRONTEND=noninteractive
# 【关键】全局允许 pip 修改系统包 (解决 Ubuntu 24.04 PEP 668 问题)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# --------------------------------------------------------------------------------
# 3. 系统配置与依赖安装
# --------------------------------------------------------------------------------


# 安装依赖 (python-is-python3 自动建立 python -> python3 软链接)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    libgl1 \
    libglib2.0-0 \
    git \
    vim\
    nano\
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------------
# 4. 关键设置：数据挂载点与软链接 (保留之前的 Offline-First 逻辑)
# --------------------------------------------------------------------------------
# 创建挂载点：/app/relbench 用于数据，/app/outputs 用于日志结果
RUN mkdir -p /app/relbench /app/outputs /root/.cache

# 创建软链接：强制定向 RelBench 的默认缓存路径到我们的挂载点
# 这样代码读取 ~/.cache/relbench 时，实际读的是挂载的硬盘
RUN ln -s /app/relbench /root/.cache/relbench

# --------------------------------------------------------------------------------
# 5. Python 环境安装
# --------------------------------------------------------------------------------
# 安装 PyTorch 2.4.0 (cu124) - 适配 H200
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 复制依赖文件 (利用 Docker 缓存层)
COPY requirements.txt /app/

# 安装项目依赖 (使用清华源加速)
# 同时安装 ultralytics (保留您的需求)
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# --------------------------------------------------------------------------------
# 6. 项目代码与入口
# --------------------------------------------------------------------------------
COPY . /app

# 赋予脚本权限
RUN chmod +x /app/relbdata/*.sh 2>/dev/null || true

# 默认入口：显示训练脚本帮助，方便用户确认环境正常
CMD ["python", "relbdata/train_on_relbench.py", "--help"]