#!/usr/bin/env bash
# 不依赖 docker compose — 用 docker run 启动全部服务 (适合 compose 下载慢/装不上的环境)
#
# 用法 (在项目根目录, 且含 Qwen3-Embedding-4B / Qwen3-Reranker-4B / Qwen3.5-9B):
#   chmod +x docs/start-services.sh
#   ./docs/start-services.sh
#
# 可选环境变量:
#   WORK_DIR        项目根目录 (默认: 脚本所在目录的上一级)
#   MILVUS_DATA_DIR Milvus 数据目录 (默认: /fs_mol/wangzhengyan/dp-rag-milvus)
#   VLLM_API_KEY    三个 vLLM 服务共用 API Key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 默认工作目录: 仓库根; 服务器上模型在 /fs_mol/wangzhengyan 时请 export WORK_DIR=/fs_mol/wangzhengyan
WORK_DIR="${WORK_DIR:-/fs_mol/wangzhengyan}"
MILVUS_DATA_DIR="${MILVUS_DATA_DIR:-/fs_mol/wangzhengyan/dp-rag-milvus}"
VLLM_API_KEY="${VLLM_API_KEY:-dp_rag_vllm_sk_8f3a9c2e1b7d4f6a}"

cd "${WORK_DIR}"

for d in Qwen3-Embedding-4B Qwen3-Reranker-4B Qwen3.5-9B; do
  if [[ ! -d "${WORK_DIR}/${d}" ]]; then
    echo "错误: 缺少模型目录 ${WORK_DIR}/${d}"
    echo "请在 WORK_DIR 下放置三个模型后再运行。"
    exit 1
  fi
done

mkdir -p "${MILVUS_DATA_DIR}"
echo "WORK_DIR=${WORK_DIR}"
echo "MILVUS_DATA_DIR=${MILVUS_DATA_DIR}"

run_if_missing() {
  local name="$1"
  shift
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "[skip] 容器已存在: ${name} (如需重建请先 docker rm -f ${name})"
    return 0
  fi
  echo "[start] ${name}"
  docker run -d --name "${name}" "$@"
}

# 1) Milvus
run_if_missing milvus-standalone \
  --restart unless-stopped \
  --security-opt seccomp=unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e COMMON_STORAGETYPE=local \
  -v "${MILVUS_DATA_DIR}:/var/lib/milvus" \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.5.4 \
  milvus run standalone

# 2) Embedding :8002
run_if_missing vllm-embed \
  --gpus all --restart unless-stopped \
  --log-opt max-size=50m --log-opt max-file=3 \
  -v "${WORK_DIR}/Qwen3-Embedding-4B:/Qwen3-Embedding-4B" \
  -p 8002:8002 \
  vllm/vllm-openai:latest \
  /Qwen3-Embedding-4B \
  --host 0.0.0.0 --port 8002 \
  --api-key "${VLLM_API_KEY}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.25 \
  --max-model-len 9216

# 3) Reranker :8001
run_if_missing vllm-rerank \
  --gpus all --restart unless-stopped \
  --log-opt max-size=50m --log-opt max-file=3 \
  -v "${WORK_DIR}/Qwen3-Reranker-4B:/models/Qwen3-Reranker-4B" \
  -p 8001:8001 \
  vllm/vllm-openai:latest \
  /models/Qwen3-Reranker-4B \
  --host 0.0.0.0 --port 8001 \
  --api-key "${VLLM_API_KEY}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.25 \
  --max-model-len 10240 \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'

# 4) LLM :8000
run_if_missing vllm-llm \
  --gpus all --restart unless-stopped \
  --log-opt max-size=50m --log-opt max-file=3 \
  -v "${WORK_DIR}/Qwen3.5-9B:/models/Qwen3.5-9B" \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /models/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --api-key "${VLLM_API_KEY}" \
  --max-model-len 30960 \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 128 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# 5) Attu (可选)
run_if_missing milvus-attu \
  --restart unless-stopped \
  -p 3000:3000 \
  zilliz/attu:v2.5.10

echo ""
echo "完成。查看状态: docker ps"
echo "验证 LLM: curl -H \"Authorization: Bearer ${VLLM_API_KEY}\" http://localhost:8000/v1/models"
