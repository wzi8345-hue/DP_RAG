FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖 (先单独 COPY 以利用构建缓存)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 项目代码: 构建上下文为 pipeline/ 目录, 须作为 `pipeline` 包拷入,
# 以保证 `uvicorn pipeline.api.app:app` 的绝对/相对导入 (from ..pipeline / ...flows) 正常工作。
COPY . /app/pipeline

# 时区 + 语言
ENV TZ=Asia/Shanghai
ENV LANG=zh_CN.UTF-8

# /app 必须在 sys.path 上, 才能 import 顶层 `pipeline` 包
ENV PYTHONPATH=/app

# 默认配置 (可通过 -v 覆盖)
ENV CONFIG_PATH=""

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

CMD ["uvicorn", "pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
