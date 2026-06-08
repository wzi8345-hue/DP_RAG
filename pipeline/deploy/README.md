# DP-RAG Pipeline 部署指南

## 架构概览

```
┌──────────────────────────────────────────────┐
│  服务器 (需 GPU)                              │
│                                               │
│  vllm-llm (:8000)    ← Qwen3.5-9B      GPU0  │
│  vllm-rerank (:8000) ← Qwen3-Reranker-4B GPU1│
│  vllm-emb (:8000)    ← Qwen3-Embedding-4B GPU1│
│                                               │
│  pipeline-api (:8080) ← FastAPI          CPU  │
│  milvus (:19530)                         CPU  │
└──────────────────────────────────────────────┘
```

**模型权重总量**: ~34GB (9B ~18GB + Reranker ~8GB + Embedding ~8GB)

---

## 前置条件

| 要求 | 说明 |
|---|---|
| GPU | 2x GPU (推荐 A100 80GB / A6000 48GB); 最低 1x 80GB |
| Docker | >= 24.0 |
| NVIDIA Container Toolkit | `nvidia-container-toolkit` 已安装 |
| 磁盘 | >= 100GB (模型权重 + Milvus 数据 + 解析产物) |
| 内存 | >= 64GB |

### 安装 NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 一、Docker Compose 部署 (推荐)

### 1. 准备模型权重

```bash
# 在宿主机上创建模型目录
sudo mkdir -p /data/models

# 将模型文件放入该目录, 确保结构如下:
/data/models/
├── Qwen3.5-9B/          # LLM
│   ├── config.json
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   └── ...
├── Qwen3-Reranker-4B/   # Reranker
│   ├── config.json
│   ├── model-*.safetensors
│   └── ...
└── Qwen3-Embedding-4B/  # Embedding
    ├── config.json
    ├── model-*.safetensors
    └── ...
```

如果模型已在其他路径, 设置环境变量:

```bash
export MODEL_DIR=/your/model/path
```

### 2. 准备数据目录

```bash
sudo mkdir -p /data/uploads /data/mineru_result /data/uniparser_result /data/milvus

# 可选: 设置数据根目录
export DATA_DIR=/data
```

### 3. 配置认证信息

```bash
# MinerU API (如需使用)
export MINERU_AUTHORIZATION="Bearer eyJ0eXAi..."

# UniParser API (如需使用)
export UNIPARSER_API_KEY="up_xxx..."

# API Key 认证 (逗号分隔; 留空则不鉴权)
export API_KEYS="key1,key2"
```

### 4. 修改 GPU 分配

编辑 `docker-compose.yml` 中的 `NVIDIA_VISIBLE_DEVICES`:

**2x GPU 配置 (推荐)**:
```yaml
vllm-llm:     NVIDIA_VISIBLE_DEVICES: "0"
vllm-rerank:  NVIDIA_VISIBLE_DEVICES: "1"
vllm-emb:     NVIDIA_VISIBLE_DEVICES: "1"
```

**1x GPU 配置 (80GB+ 显存)**:
```yaml
vllm-llm:     NVIDIA_VISIBLE_DEVICES: "0"   # 独占
vllm-rerank:  NVIDIA_VISIBLE_DEVICES: "0"   # 共享
vllm-emb:     NVIDIA_VISIBLE_DEVICES: "0"   # 共享
```
注意: 同卡跑 3 个 vLLM 时, LLM 需降低 `--gpu-memory-utilization` 到 0.60 左右。

### 5. 启动所有服务

```bash
cd deploy/

# 构建并启动 (首次需要下载 vLLM/Milvus 镜像, 可能需要 10-20 分钟)
docker compose up -d

# 查看日志
docker compose logs -f pipeline-api

# 检查各服务状态
docker compose ps
```

### 6. 验证服务

```bash
# 健康检查
curl http://localhost:8080/api/v1/health

# 单次查询
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MoS2的晶格常数是多少?"}'

# 多轮对话
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "这篇论文研究了什么?"}'

# 流式对话
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "介绍MoS2的基本性质"}'

# 集合统计
curl http://localhost:8080/api/v1/stats
```

### 7. 停止服务

```bash
docker compose down          # 停止, 数据保留
docker compose down -v       # 停止并删除 Milvus 数据 (慎用)
```

---

## 二、非 Docker 部署 (systemd)

适合: 不想用 Docker, 或需要更细粒度控制。

### 1. 安装 Python 依赖

```bash
cd /opt/dp-rag/pipeline
conda create -n dp-rag python=3.12 -y
conda activate dp-rag
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务

```bash
# LLM (终端 1)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.5-9B \
  --port 8000 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code

# Reranker (终端 2)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3-Reranker-4B \
  --port 8001 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code

# Embedding (终端 3)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3-Embedding-4B \
  --port 8002 \
  --gpu-memory-utilization 0.45 \
  --trust-remote-code
```

### 3. 启动 Milvus (Docker 仍推荐)

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v /data/milvus:/var/lib/milvus \
  milvusdb/milvus:v2.4.17
```

### 4. 启动 Pipeline API

```bash
# 使用默认配置 (localhost)
uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8080

# 或使用自定义配置
CONFIG_PATH=/opt/dp-rag/prod_config.yaml uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8080
```

### 5. 注册 systemd 服务 (生产环境)

创建 `/etc/systemd/system/dp-rag-api.service`:

```ini
[Unit]
Description=DP-RAG Pipeline API
After=network.target

[Service]
Type=simple
User=dp
WorkingDirectory=/opt/dp-rag/pipeline
Environment=CONFIG_PATH=/opt/dp-rag/prod_config.yaml
Environment=PATH=/opt/conda/envs/dp-rag/bin:/usr/bin
ExecStart=/opt/conda/envs/dp-rag/bin/uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable dp-rag-api
sudo systemctl start dp-rag-api
sudo journalctl -u dp-rag-api -f   # 查看日志
```

---

## API 参考

### 查询

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/query` | 单次查询 |
| POST | `/api/v1/chat` | 多轮对话 (返回完整结果) |
| POST | `/api/v1/chat/stream` | SSE 流式对话 |
| POST | `/api/v1/sessions` | 创建对话会话 |
| DELETE | `/api/v1/sessions/{id}` | 销毁会话 |

### 灌入 (异步)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/ingest/rebuild` | 全量重灌 (清空后重灌) |
| POST | `/api/v1/ingest/append` | 增量追加 |
| POST | `/api/v1/ingest/parse` | 仅解析 PDF |
| POST | `/api/v1/ingest/load-vec` | 直接灌入已向量化文件 |
| GET | `/api/v1/tasks/{id}` | 查询异步任务状态 |

### 文件与运维

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/files/upload` | 上传 PDF |
| GET | `/api/v1/stats` | Milvus 集合统计 |
| GET | `/api/v1/health` | 健康检查 |

### 请求示例

**多轮对话**:
```bash
# 1. 创建会话
SESSION_ID=$(curl -s -X POST http://localhost:8080/api/v1/sessions | jq -r '.session_id')

# 2. 第一轮
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"MoS2是什么?\", \"session_id\": \"$SESSION_ID\"}"

# 3. 第二轮 (带上下文)
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"它的带隙呢?\", \"session_id\": \"$SESSION_ID\"}"
```

**流式对话**:
```bash
curl -N -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "介绍MoS2的基本性质"}'
```

**异步灌入**:
```bash
# 提交任务
TASK_ID=$(curl -s -X POST http://localhost:8080/api/v1/ingest/append \
  -H "Content-Type: application/json" \
  -d '{"directory": "/app/mineru_result"}' | jq -r '.id')

# 查询进度
curl http://localhost:8080/api/v1/tasks/$TASK_ID
```

**SSE 事件格式**:
```
data: {"type": "status", "stage": "retrieving"}

data: {"type": "status", "stage": "generating"}

data: {"type": "text", "content": "MoS2"}

data: {"type": "text", "content": "是一种"}

data: {"type": "done", "answer": "MoS2是一种...", "hits": [...], "latency_s": 3.2, "session_meta": {...}}
```

---

## 常见问题

### 1. vLLM 启动慢 / OOM

- 首次启动需要加载模型, 30-120 秒属正常
- OOM: 降低 `--gpu-memory-utilization` 或换小模型
- 同卡多 vLLM: 总 `gpu-memory-utilization` 之和应 < 0.95

### 2. Milvus 连接失败

- 检查容器状态: `docker compose ps milvus`
- 查看日志: `docker compose logs milvus`
- 等 Milvus 健康后再启动 API (compose 的 `depends_on: condition: service_healthy` 已处理)

### 3. 流式输出中断

- Nginx 反代时需加 `proxy_buffering off;`
- 加 `X-Accel-Buffering: no` header (已在代码中设置)
- 检查超时设置: `proxy_read_timeout 300s;`

### 4. 配置覆盖

- `default_config.yaml` 是默认值, 不要直接改
- 生产配置通过 `prod_config.yaml` 覆盖
- 敏感信息 (api_key) 通过环境变量注入: `${MINERU_AUTHORIZATION}`

### 5. 模型 dim 不匹配

修改 `embedding.model` 后, Milvus 集合的 `dim` 必须同步修改, 并 rebuild:
```
Qwen3-Embedding-0.6B → dim: 1024
Qwen3-Embedding-4B   → dim: 2560
Qwen3-Embedding-8B   → dim: 4096
```
