# 1. 基础镜像（轻量级 Python 3.10）
FROM python:3.10-slim

# 2. 设置工作目录（容器里的文件夹）
WORKDIR /app

# 3. 先复制依赖文件（利用 Docker 缓存，加快构建速度）
COPY requirements.txt .

# 4. 安装依赖（只安装必要的，镜像越小越好）
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制当前目录所有代码到容器里
COPY . .

# 6. 暴露端口（和外部通信的门）
EXPOSE 8000

# 7. 启动命令（容器跑起来时执行的指令）
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]