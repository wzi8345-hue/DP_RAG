"""运行前在此填写 LLM 与生成参数，然后直接运行 gen_qa_dataset.py。"""

# ---------- LLM（必填）----------
API_BASE = "https://api.gpugeek.com/v1"  # OpenAI 兼容接口地址
MODEL = "Vendor3/qwen3.5-plus"
API_KEY = "00ilsqru12sxx301000di4oxj0ne7mx800n84hks"  # 在此填写，或设置环境变量 LLM_API_KEY

# ---------- 数据路径 ----------
# 默认读取上级目录 DP_rag/mineru_result；可改为绝对路径
MINERU_RESULT_DIR = None

# ---------- 生成选项 ----------
# 全量 mineru 切块文献 → test_dataset_all.json; 仅检索库子集 → test_dataset_corpus33.json
OUTPUT_FILE = "test_dataset_首钢文献.json"
SKIP_DIRS = ["测试", "测试2"]
DOC_FILTER = None  # 只处理目录名包含该关键词的文献，None 表示全部
CONTINUE_FROM = None  # 续跑：已有结果文件名，如 "test_dataset.json"

# ---------- 文献范围 ----------
# "all"        → mineru_result 下所有含 knowledge_blocks.json 的目录 (默认, 632 篇)
# "in_corpus"  → 仅 Milvus 检索库中已灌入的文献 (可立即评测的子集)
DATASET_SCOPE = "all"
# 是否要求 knowledge_blocks_vec.json 也存在 (chunk+向量化都完成)
REQUIRE_VECTORIZED = True

# ---------- 评测可用性 (legacy; 被 DATASET_SCOPE 覆盖) ----------
IN_CORPUS_ONLY = False
# Milvus-lite 数据库路径; None 表示用上级目录的 milvus_lite.db
MILVUS_DB = None
