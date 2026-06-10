# Milvus 数据格式说明

> 本文档描述 DP-RAG 灌入 Milvus 时的 **Collection Schema**、字段含义、字段来源，以及来自本仓库已上传数据的真实样例。
>
> - **Schema 版本**：v5（dense + BM25 sparse 混合检索）
> - **默认 Collection**：`literature_chunks`
> - **自定义知识库 Collection**：`kb_<8位hex>`（例如 `kb_a847ccfd`）
> - **向量维度**：由 Embedding 模型决定；当前配置为 **2560**（`Qwen3-Embedding-4B`）

---

## 目录

1. [数据流概览](#1-数据流概览)
2. [Collection 与命名规则](#2-collection-与命名规则)
3. [Schema 字段定义](#3-schema-字段定义)
4. [字段来源说明](#4-字段来源说明)
5. [真实数据样例](#5-真实数据样例)
6. [检索相关说明](#6-检索相关说明)
7. [重新灌入（一致性保证）](#7-重新灌入一致性保证)
8. [多用户隔离设计（规划）](#8-多用户隔离设计规划)

---

## 1. 数据流概览

```
PDF 上传
  → 解析（UniParser / MinerU）
  → 切块 → knowledge_blocks.json
  → 向量化 → knowledge_blocks_vec.json
  → chunk_to_row() 字段映射
  → MilvusIngester.ingest_file() 批量写入
  → Collection（literature_chunks 或 kb_*）
```

本地中间产物路径示例：

```
uploads/kb_a847ccfd/测试/knowledge_blocks.json
uploads/kb_a847ccfd/测试/knowledge_blocks_vec.json
```

写入 Milvus 时，`sparse_embedding` **不需要手动提供**，由 Milvus BM25 Function 根据 `embedding_text` 自动生成。

---

## 2. Collection 与命名规则

| Collection 名称 | 用途 | 示例 |
|---|---|---|
| `literature_chunks` | 系统默认知识库 | 批量导入的文献库 |
| `kb_<id>` | 用户自建知识库 | `kb_a847ccfd`（名称「材料科学」） |

**文档元数据推断规则**（`infer_doc_metadata`）：

| 字段 | 优先级 |
|---|---|
| `doc_id` | 显式参数 > sidecar `*_meta.json` > 文件名（去 `_vec` 后缀） |
| `doc_name` | 显式参数 > sidecar `doc_name` > 文件名；UniParser 路径会优先使用论文真实标题 |
| `publication_year` | 显式参数 > sidecar > 文件名年份 > chunk 内容扫描 > `0` |

**主键 `pk`** 固定为：`{doc_id}::{chunk_id}`，例如 `测试::title_bd717e29`。

---

## 3. Schema 字段定义

| 字段 | Milvus 类型 | 最大长度 / 维度 | 说明 |
|---|---|---|---|
| `pk` | `VARCHAR`（主键） | 256 | `{doc_id}::{chunk_id}` |
| `doc_id` | `VARCHAR` | 128 | 文档 ID，通常为 PDF 文件名（无扩展名）或知识库目录名 |
| `doc_name` | `VARCHAR` | 512 | 展示用文档名（可为论文标题） |
| `chunk_id` | `VARCHAR` | 64 | 块 ID，如 `text_d36254f6` |
| `type` | `VARCHAR` | 16 | 块类型，见下表 |
| `section` | `VARCHAR` | 1024 | 章节标题 |
| `page_start` | `INT32` | — | 起始页码（0-based）；无页码时为 `-1` |
| `paragraph_index` | `INT32` | — | 正文段落序号（1-based）；`-1` 非正文；`0` 为 LLM 摘要 |
| `publication_year` | `INT32` | — | 发表年份；未知为 `0` |
| `content` | `VARCHAR` | 32000 | 块正文 |
| `context` | `VARCHAR` | 8000 | 附加上下文（公式脚注、表格说明等） |
| `related_assets` | `JSON` | — | 关联图表/公式/参考文献的交叉引用 |
| `embedding_text` | `VARCHAR`（启用中文分词） | 32000 | 用于 dense 向量 + BM25 稀疏检索的拼接文本 |
| `embedding` | `FLOAT_VECTOR` | `dim`（当前 2560） | Dense 向量，由 Embedding 模型生成 |
| `sparse_embedding` | `SPARSE_FLOAT_VECTOR` | — | BM25 稀疏向量，**入库时自动生成** |
| `owner` | `VARCHAR` | 64 | **（规划 v6，当前未启用）** 数据归属用户 ID。多用户硬隔离的**行级兜底过滤字段**，整篇文档/整个集合恒定，入库时由登录用户注入。详见 [第 8 节](#8-多用户隔离设计规划) |

### 3.1 `type` 合法取值

| 值 | 含义 |
|---|---|
| `title` | 文档标题块 |
| `summary` | LLM 合成的摘要块（`paragraph_index = 0`） |
| `text` | 正文段落 |
| `table` | 表格块 |
| `image` | 图片块（语义来自 Caption + context） |
| `equation` | 独立公式块 |
| `references` | 参考文献聚合块 |

### 3.2 `related_assets` 结构

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

## 4. 字段来源说明

一条记录（`chunk_to_row()` 产出）的字段有三类来源：直接映射、入库时注入、Milvus 自动生成。

**① 直接来自向量化产物 `knowledge_blocks_vec.json`**（部分字段改名）：

| 记录字段 | 来源字段 | 说明 |
|---|---|---|
| `chunk_id` | `id` | 改名 |
| `page_start` | `pages` | 取 `pages[0]`，无页码为 `-1` |
| `type` / `section` / `paragraph_index` / `content` / `context` / `related_assets` / `embedding_text` / `embedding` | 同名 | 原样映射 |

**② 入库时注入**（文档级，整篇文档所有 chunk 相同）：

| 记录字段 | 来源 |
|---|---|
| `pk` | 拼接 `{doc_id}::{chunk_id}` |
| `doc_id` / `doc_name` / `publication_year` | 由 `infer_doc_metadata()` 推断（见 [第 2 节](#2-collection-与命名规则)） |

**③ Milvus 自动生成**：

| 记录字段 | 来源 |
|---|---|
| `sparse_embedding` | BM25 Function 从 `embedding_text` 自动生成 |

> 向量化产物中的 `embedding_model`、`embedding_dim` 仅用于本地追溯，不进入 Milvus。

---

## 5. 真实数据样例

以下样例来自本仓库已上传的知识库数据：

- **Collection**：`kb_a847ccfd`
- **源文件**：`uploads/kb_a847ccfd/测试/knowledge_blocks_vec.json`
- **文档**：`测试.pdf`（`doc_id = 测试`，共 **117** 条 chunk）
- **块类型分布**：`title×1`, `summary×1`, `text×73`, `equation×27`, `table×6`, `references×9`

### 5.1 样例一：title 块

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

### 5.2 样例二：text 正文块

```json
{
  "pk": "测试::text_d36254f6",
  "doc_id": "测试",
  "doc_name": "测试",
  "chunk_id": "text_d36254f6",
  "type": "text",
  "section": "Structural, Electronic, Mechanical, Thermodynamic, and Linear and Nonlinear Optical Properties of ${ \\mathsf { MoS } } _ { 2 }$ , ${ \\mathsf { MoSe } } _ { 2 }$ , and their $\\mathsf { MoS } _ { 2 x } \\mathsf { S e } _ { 2 ( 1 - x ) }$ Alloys: Ab Initio Calculations",
  "page_start": 0,
  "paragraph_index": 1,
  "publication_year": 0,
  "content": "Y. ASADI1 and Z. NOURBAKHSH1,2 1.—Department of Physics, Faculty of Sciences, University of Isfahan, Isfahan, Iran. 2.—e-mail: z.nourbakhsh@sci.ui.ac.ir",
  "context": "",
  "related_assets": [],
  "embedding_text": "[Section] Structural, Electronic, Mechanical, Thermodynamic, and Linear and Nonlinear Optical Properties of ${ \\mathsf { MoS } } _ { 2 }$ , ${ \\mathsf { MoSe } } _ { 2 }$ , and their $\\mathsf { MoS } _ { 2 x } \\mathsf { S e } _ { 2 ( 1 - x ) }$ Alloys: Ab Initio Calculations\n\nY. ASADI1 and Z. NOURBAKHSH1,21.—Department of Physics, Faculty of Sciences, University of Isfahan, Isfahan, Iran.2.—e-mail: z.nourbakhsh@sci.ui.ac.ir",
  "embedding": "<float[2560]>",
  "sparse_embedding": "<由 BM25 Function 从 embedding_text 自动生成>"
}
```

> 说明：`embedding` 数组在文档中省略具体数值以节省篇幅，实际为 2560 维浮点向量；`sparse_embedding` 由 Milvus BM25 Function 从 `embedding_text` 自动生成。

---

## 6. 检索相关说明

- **Dense 检索**：对 query 做 Embedding 后，在 `embedding` 字段上做向量相似度搜索（默认内积 `IP`）。
- **Sparse 检索（BM25）**：对 query 文本，在 `sparse_embedding` 上做 BM25 全文检索；`embedding_text` 字段启用了中文分词 analyzer。
- **混合检索**：系统默认将 dense 与 sparse 结果融合（具体权重见 `pipeline/default_config.yaml` 中 `retrieval` 配置）。
- **过滤**：可按 `doc_id`、`type`、`publication_year` 等标量字段做 Milvus filter 表达式。

### 6.1 从 Milvus 直接查询样例（Python）

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

---

## 7. 重新灌入（一致性保证）

当 Milvus 需要重灌（换机器、集合损坏、误删等），可以**直接复用本地已落盘的 `knowledge_blocks_vec.json`**，重灌结果与首次入库**逐字节一致**，且**不依赖 embedding / summary LLM 服务在线**（不会重新 chunk、不会重新 embed，向量原样喂回）。

### 8.1 可重放单元：vec.json + meta.json

每篇文档目录下有两个关键文件，二者构成一个自包含、可重放的单元：

```
uploads/kb_a847ccfd/测试/
  knowledge_blocks_vec.json     # 每条 chunk + 向量
  knowledge_blocks_meta.json    # 文档级元数据（入库时落盘）
```

`knowledge_blocks_meta.json`（sidecar）由入库时（`MilvusIngester.ingest_file()`）写入，记录本次入库实际使用的文档级字段：

```json
{
  "doc_id": "测试",
  "doc_name": "Structural, Electronic, ... MoS2 Alloys: Ab Initio Calculations",
  "publication_year": 0
}
```

> 这三个字段不在 vec.json 里，所以单靠 vec.json 无法还原 `pk` / `doc_id` / `doc_name`。sidecar 补齐了它们，使重灌可以复现完全相同的 Milvus 行。UniParser 路径的 sidecar 还会包含 `source` / `filename` / 页数等附加信息。

### 8.2 还原优先级

重灌时 `doc_id` / `doc_name` / `publication_year` 的来源：

| 字段 | 重灌时来源 |
|---|---|
| `doc_id` | 显式传入文档目录名（与首次一致） > sidecar `doc_id` > vec 文件名 |
| `doc_name` | sidecar `doc_name`（含 UniParser 真实标题） > 目录名 |
| `publication_year` | sidecar `publication_year` > 文件名年份 > 内容扫描 > `0` |

### 8.3 重灌方式

**方式一：知识库「重建」按钮 / API（推荐）**

```
POST /api/v1/collections/{name}/rebuild
```

后台会扫描 `uploads/<kb>/**/knowledge_blocks_vec.json`，清空集合后逐字节重灌；缺 vec.json 但仍有解析产物（`uniparser_result.json` / `content_list_v2.json`）的文档，自动回退到完整 `chunk → embed → store`。

**方式二：命令行直接重放单篇 vec.json**

```python
from pipeline.clients.milvus import MilvusIngester

ingester = MilvusIngester(
    uri="http://localhost:19530",
    collection="kb_a847ccfd",
    dim=2560,
    recreate=False,   # True = 先清空整个集合
)
ingester.ingest_file(
    "uploads/kb_a847ccfd/测试/knowledge_blocks_vec.json",
    doc_id="测试",      # 传文档目录名，保证 pk 命名空间与首次一致
    purge_existing=True,
)
```

> ⚠️ 不要省略 `doc_id`。直接对 `knowledge_blocks_vec.json` 调用且无 sidecar 时，`doc_id` 会从文件名推断成 `knowledge_blocks`，导致所有文档撞进同一个 `pk` 命名空间。传目录名（或依赖 sidecar）可避免。

---

## 8. 多用户隔离设计（规划）

> **现状**：系统暂无登录/账号体系，数据隔离完全靠 **Collection（知识库）边界**——每个 `kb_*` 集合即一个独立知识库，互不可见。
>
> **目标**：引入"用户登录 + 每个用户只能看/检索自己的知识库"的硬隔离。
>
> ⚠️ 本节为**设计规划**：当前 v5 schema **未包含** `owner` 字段，前后端也**未接入**登录体系。落地需配合后端用户体系，届时升级到 v6。

### 8.1 为什么不把 `user` 加到每条 chunk 做隔离

在本系统里"**一个知识库整体属于一个用户**"，隔离的天然边界是 **Collection**，而不是向量行。给每条 chunk 塞一个 user 字段来做隔离是冗余的（同一集合里所有行的 user 都一样），还会带来"每次检索都得记得加过滤、漏一次就串号"的风险。因此归属信息应挂在**知识库**上，隔离在**集合**层完成。

### 8.2 两层隔离

**第一层（必需）：知识库归属用户 = 集合归属用户**

- 每个 `kb_*` 集合整体归属一个用户；归属信息存在知识库元数据 `.kb_meta.json` 增加 `owner` 字段（用户 ID）。
- 登录后，后端**只列出/检索该用户 `owner` 名下的集合**。这一层即实现硬隔离——用户根本拿不到别人集合的句柄。
- 集合命名可选加用户前缀（如 `kb_<uid8>_<hash>`）便于运维辨认，但**不依赖名字做鉴权**。

**第二层（推荐，兜底防泄漏）：每条 Milvus 行冗余 `owner`**

- schema 增加 `owner VARCHAR(64)`，**整集合恒定**，入库时由登录用户注入（与 `doc_id` / `doc_name` 同属"入库时注入"类字段）。
- 每次检索**强制附加** `owner == "<当前用户>"` 过滤表达式。即便集合层出现命名/路由 bug，也不会把别人的数据搜出来（行级安全兜底）。
- 代价：每行多一个短字符串，可忽略；好处是数据自描述、未来若做跨库检索/共享库也无需再迁移。

### 8.3 字段来源与注入点（规划）

| 字段 | 来源 |
|---|---|
| `owner`（KB 级，`.kb_meta.json`） | 创建知识库时的登录用户 |
| `owner`（Milvus 行级） | 入库时由后端从登录态注入（`chunk_to_row()` 增加该字段） |

### 8.4 存量迁移

- 存量数据无 `owner`：可设默认归属（如 `owner = "legacy"` 或某管理员账号），或按现有 `.kb_meta.json` 回填。
- **主键 `pk` 不变**，仍为 `{doc_id}::{chunk_id}`——集合已隔离命名空间，无需把 user 并入 pk。

### 8.5 结论

| 诉求 | 建议 |
|---|---|
| 只是"每个用户看自己的库" | **第一层（KB 级 `owner`）即足够，不必动 Milvus schema** |
| 对隔离安全要求高（多用户共享一套部署） | 再叠加**第二层**（行级 `owner` + 强制过滤）作为兜底 |

---

## 参考代码

| 模块 | 路径 | 职责 |
|---|---|---|
| Schema 定义 | `pipeline/clients/milvus.py` → `build_schema()` | 创建 v5 Collection |
| 字段映射 | `pipeline/clients/milvus.py` → `chunk_to_row()` | chunk → Milvus 行 |
| 元数据推断 | `pipeline/clients/milvus.py` → `infer_doc_metadata()` | doc_id / doc_name / year |
| Embedding 文本 | `pipeline/processors/vectorizer.py` → `compose_embedding_text()` | 生成 `embedding_text` |
| sidecar 写入 | `pipeline/clients/milvus.py` → `_write_meta_sidecar()` | 入库时落 `*_meta.json`（可重放） |
| 复用向量重灌 | `pipeline/flows/ingest.py` → `reingest_from_directory()` | 直灌 vec.json，不重新 embed |
| 默认配置 | `pipeline/default_config.yaml` → `milvus` / `embedding` | 连接、维度、索引参数 |
