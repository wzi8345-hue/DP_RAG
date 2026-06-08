# RAGAS 评估 - DP-RAG Pipeline

使用 [RAGAS](https://docs.ragas.io/) 框架评估 DP-RAG Pipeline 的检索与生成质量。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. (可选) 自动生成测试数据集
python generate_dataset.py --num 30 --output datasets/my_dataset.json

# 3. 运行评估
python evaluate.py
```

## 评估模式

| 模式 | 说明 | 指标 |
|------|------|------|
| `retrieval` | 仅评估检索质量 | Context Precision, Context Recall |
| `full` | 检索 + 生成全链路 | 上述 + Faithfulness, Answer Relevancy |

```bash
# 仅评估检索
python evaluate.py --mode retrieval

# 全链路评估 (含生成质量)
python evaluate.py --mode full
```

## 指标说明

| 指标 | 含义 | 范围 |
|------|------|------|
| **Context Precision** | 检索结果中相关 chunk 的比例 (排在前面的越相关得分越高) | 0~1, 越高越好 |
| **Context Recall** | 回答问题所需的信息是否都被检索到了 | 0~1, 越高越好 |
| **Faithfulness** | 生成答案是否严格基于检索上下文 (无幻觉) | 0~1, 越高越好 |
| **Answer Relevancy** | 生成答案与问题的相关程度 | 0~1, 越高越好 |

## 数据集格式

数据集为 JSON 数组, 每条记录包含以下字段:

```json
[
  {
    "question": "用户问题",
    "ground_truth": "标准参考答案",
    "ground_contexts": ["与答案相关的关键句1", "关键句2"]
  }
]
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `question` | 是 | 用户查询问题 |
| `ground_truth` | 是 | 标准参考答案, 用于 Context Recall 和 Faithfulness 评估 |
| `ground_contexts` | 否 | 问题相关的关键句列表, 用于辅助评估 (可选) |

### 手动创建数据集

1. 复制示例文件: `cp datasets/test_dataset.json datasets/my_dataset.json`
2. 根据你的文献库内容编写 question + ground_truth 对
3. 修改 `config.yaml` 中的 `dataset` 指向你的数据集

### 自动生成数据集

```bash
# 默认生成 20 条
python generate_dataset.py

# 指定数量和输出路径
python generate_dataset.py --num 50 --output datasets/large_dataset.json

# 只从 table/image 类型生成
python generate_dataset.py --types table image
```

自动生成会从 Milvus 中随机采样 chunk, 调用 LLM 生成 question + ground_truth 对。

## 配置说明

编辑 `config.yaml`:

```yaml
# 评估模式
mode: "retrieval"       # retrieval / full

# 测试数据集路径
dataset: "datasets/test_dataset.json"

# RAGAS 评估用的 LLM (用于计算 faithfulness 等指标)
ragas_llm:
  api_base: "https://api.gpugeek.com/v1"
  model: "Vendor2/GPT-5-mini"
  api_key: "your-api-key"

# RAGAS 评估用的 Embeddings (用于 answer_relevancy 指标)
ragas_embeddings:
  api_base: "https://your-embed-endpoint/v1"
  model: "model"
  api_key: "your-api-key"

# 评估指标
metrics:
  - context_precision
  - context_recall
```

## 输出结果

评估完成后在 `results/` 目录下生成:

- `metrics_report.json` — 完整结果 (含每条问题的详细数据)
- `metrics_report.csv` — 指标汇总表

示例输出:

```
============================================================
  RAGAS 评估结果 (mode=retrieval)
============================================================
  context_precision: 0.7200
  context_recall: 0.6500
  成功: 20/20
  结果已保存到: results/
============================================================
```

## 目录结构

```
ragas_eval/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── config.yaml                # 评估配置
├── evaluate.py                # 主评估脚本
├── generate_dataset.py        # 数据集自动生成
├── datasets/
│   ├── test_dataset.json      # 手写示例数据集 (5 条)
│   └── auto_generated.json    # 自动生成数据集 (运行后)
└── results/                   # 评估结果输出 (运行后)
    ├── metrics_report.json
    └── metrics_report.csv
```
