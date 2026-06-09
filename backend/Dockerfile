# DP-RAG 后端镜像：uv 管理依赖 + FastAPI + RAG pipeline。
FROM python:3.11-slim

# uv（依赖与虚拟环境管理）
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 系统依赖（部分解析/向量库客户端需要 build 工具；psycopg[binary] 自带 libpq）
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用层缓存。有 uv.lock 用 --frozen，否则即时解析。
COPY pyproject.toml uv.lock* ./
RUN if [ -f uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi

# 拷贝后端代码
COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${API_PORT}/api/v1/health" || exit 1

# 镜像默认不加载 local_api_config.yaml（那是本机联调用）。
# 容器内基建连接走环境变量覆盖（EMBEDDING_API_BASE / MILVUS_URI / LLM_API_BASE …）。
# 如需用 YAML，运行时设 CONFIG_PATH 并挂载文件即可（环境变量优先级更高）。
ENV CONFIG_PATH=""

CMD ["python", "run_api.py"]
