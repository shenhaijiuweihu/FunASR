# Dockerfile
FROM python:3.10-slim

# --- ★★★ 核心优化点 ★★★ ---
# 切换到国内的 Debian 软件源来加速 apt-get
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# 设置工作目录
WORKDIR /app

# 安装系统级的编译工具
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# --- ★★★ 核心优化点 ★★★ ---
# 切换到国内的 PyPI 镜像源来加速 pip install
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U yt-dlp
RUN pip install funasr gunicorn torch torchaudio
# 复制你的应用代码
COPY video.py .

# 声明容器将监听的端口
EXPOSE 8000

# 容器启动时运行的命令
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "video:app"]