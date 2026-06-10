# Milvus 数据格式说明

> 本文档描述灌入 Milvus 的 **Collection Schema** 与字段含义。
>
> - **Schema 版本**：v5（dense + BM25 sparse 混合检索）
> - **向量维度**：由 Embedding 模型决定；当前为 **2560**（`Qwen3-Embedding-4B`）

---

## 1. Collection 命名

| Collection 名称 | 用途 |
|---|---|
| `literature_chunks` | 默认知识库 |
| `kb_<id>` | 自建知识库，例如 `kb_a847ccfd` |

**主键 `pk`** 固定为 `{doc_id}::{chunk_id}`，例如 `测试::title_bd717e29`。

---

## 2. Schema 字段定义

| 字段 | Milvus 类型 | 最大长度 / 维度 | 说明 |
|---|---|---|---|
| `pk` | `VARCHAR`（主键） | 256 | `{doc_id}::{chunk_id}` |
| `doc_id` | `VARCHAR` | 128 | 文档 ID（通常为文件名，无扩展名） |
| `doc_name` | `VARCHAR` | 512 | 展示用文档名（可为论文标题） |
| `chunk_id` | `VARCHAR` | 64 | 块 ID，如 `text_d36254f6` |
| `type` | `VARCHAR` | 16 | 块类型，见 [2.1](#21-type-合法取值) |
| `section` | `VARCHAR` | 1024 | 章节标题 |
| `page_start` | `INT32` | — | 起始页码（0-based）；无页码为 `-1` |
| `paragraph_index` | `INT32` | — | 正文段落序号（1-based）；`-1` 非正文；`0` 为摘要 |
| `publication_year` | `INT32` | — | 发表年份；未知为 `0` |
| `content` | `VARCHAR` | 32000 | 块正文 |
| `context` | `VARCHAR` | 8000 | 附加上下文（公式脚注、表格说明等） |
| `related_assets` | `JSON` | — | 关联图表/公式/参考文献的交叉引用 |
| `embedding_text` | `VARCHAR`（启用中文分词） | 32000 | 用于 dense 向量 + BM25 稀疏检索的拼接文本 |
| `embedding` | `FLOAT_VECTOR` | `dim`（当前 2560） | Dense 向量 |
| `sparse_embedding` | `SPARSE_FLOAT_VECTOR` | — | BM25 稀疏向量，**入库时由 BM25 Function 从 `embedding_text` 自动生成** |

### 2.1 `type` 合法取值

| 值 | 含义 |
|---|---|
| `title` | 文档标题块 |
| `summary` | 摘要块（`paragraph_index = 0`） |
| `text` | 正文段落 |
| `table` | 表格块 |
| `image` | 图片块（语义来自 Caption + context） |
| `equation` | 独立公式块 |
| `references` | 参考文献聚合块 |

### 2.2 `related_assets` 结构

```json
[
  {
    "type": "image",
    "label": "Fig. 1",
    "chunk_id": "image_a1b2c3d4"
  }
]
```

---

## 3. 数据样例

### 3.1 title 块

```json
{
  "pk": "测试::title_bd717e29",
  "doc_id": "测试",
  "doc_name": "测试",
  "chunk_id": "title_bd717e29",
  "type": "title",
  "section": "",
  "page_start": 0,
  "paragraph_index": -1,
  "publication_year": 0,
  "content": "测试",
  "context": "",
  "related_assets": [],
  "embedding_text": "测试",
  "embedding": "<float[2560]>",
  "sparse_embedding": "<由 BM25 Function 从 embedding_text 自动生成>"
}
```

### 3.2 text 正文块

```json
{
  "pk": "测试::text_d36254f6",
  "doc_id": "测试",
  "doc_name": "测试",
  "chunk_id": "text_d36254f6",
  "type": "text",
  "section": "Ab Initio Calculations of MoS2 ...",
  "page_start": 0,
  "paragraph_index": 1,
  "publication_year": 0,
  "content": "Y. ASADI and Z. NOURBAKHSH, Department of Physics, University of Isfahan, Iran.",
  "context": "",
  "related_assets": [],
  "embedding_text": "[Section] Ab Initio Calculations of MoS2 ...\n\nY. ASADI and Z. NOURBAKHSH ...",
  "embedding": "<float[2560]>",
  "sparse_embedding": "<由 BM25 Function 从 embedding_text 自动生成>"
}
```

> `embedding` 为 2560 维浮点向量，样例中省略具体数值。

---

## 4. 检索说明

- **Dense 检索**：对 query 做 Embedding 后，在 `embedding` 字段上做向量相似度搜索（默认内积 `IP`）。
- **Sparse 检索（BM25）**：对 query 文本，在 `sparse_embedding` 上做 BM25 全文检索；`embedding_text` 启用了中文分词 analyzer。
- **混合检索**：默认将 dense 与 sparse 结果融合。
- **过滤**：可按 `doc_id`、`type`、`publication_year` 等标量字段做 Milvus filter 表达式。

### 查询样例（Python）

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
rows = client.query(
    collection_name="kb_a847ccfd",
    filter='doc_id == "测试"',
    output_fields=[
        "pk", "doc_id", "doc_name", "chunk_id", "type",
        "section", "page_start", "content", "embedding_text",
    ],
    limit=5,
)
```
