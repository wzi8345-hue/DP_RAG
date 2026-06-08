# DP-RAG

面向科研文献的端到端 **Agentic RAG** 系统：从 PDF 解析、知识分块、向量化入库，到多路径检索与答案生成，并提供完整的 Web 管理界面。

```
                         ┌─────────────── 灌入链路 ───────────────┐
   PDF ──► 解析(MinerU/UniParser) ──► 知识分块 ──► 向量化 ──► Milvus 存储
                                                                    │
                         ┌─────────────── 查询链路 ───────────────┐ │
   用户问题 ──► 检索 (Hybrid / Agentic / 专家模式) ──► LLM 生成 ──► 答案(可溯源)
```

系统由三部分组成：**算法流水线（pipeline）**、**后端 API（FastAPI）**、**前端界面（React）**。

---

## 一、算法部分（`pipeline/`）

科研文献 RAG 的核心流水线，覆盖灌入与查询两条链路。

### 灌入链路（ingest）

`PDF → parse → chunk → embed → store`，每篇文档的中间产物按知识库隔离落盘，便于管理与重建。

- **解析**：支持 MinerU 与 UniParser 两种后端，将 PDF 转为结构化内容块（标题/正文/图表/引用）。
- **分块**：按标题聚合的知识块构建，自动识别摘要类章节（abstract / summary / introduction）。
- **向量化**：统一 Embedding 客户端，输出可直接入库的向量化 JSON。
- **存储**：写入 Milvus 集合，按 `doc_id` 去重，支持全量重建 / 增量追加。

### 查询链路（query）

提供三档检索能力，逐级增强：

| 模式 | 说明 |
|------|------|
| **基础检索** | `hybrid`（元数据 + 向量，RRF 融合）/ `vector` / `metadata` |
| **Agentic RAG** | 基于 LangGraph 的 LLM 路由决策 + 多路径并行检索（summary / progressive / local / metadata）+ 重排 + 反思 |
| **专家模式** | 多轮递进式文献研究：规划子问题 → 迭代检索累积证据 → 充分性判定 → 综述综合，并由「技能（Skill）」按查询类型定制规划/策略/综述提示词 |

关键检索组件位于 `pipeline/retrieval/`（`langgraph_agent.py`、`research_agent.py`、`retrievers.py`、重排与反思等），技能路由位于 `pipeline/routing/`。

### 目录结构

```
pipeline/
├── pipeline.py        # Pipeline 编排器
├── run.py             # CLI 入口
├── config.py          # 集中式配置 (默认 + 用户 + 环境变量三层覆盖)
├── default_config.yaml
├── clients/           # MinerU / UniParser / Embedding / LLM / Milvus 客户端
├── processors/        # 分块 (chunker) 与向量化 (vectorizer)
├── retrieval/         # 检索系统 (基础 / Agentic / 专家研究 / 重排 / 反思)
├── routing/           # 技能路由与研究规划
├── steps/             # Pipeline 单步 (parse/chunk/embed/store/retrieve/generate)
├── flows/             # 高级编排 (ingest / query)
├── skills/            # 内置专家技能定义 (markdown)
└── api/               # 后端 API (见下)
```

### 命令行用法

```bash
# 灌入单个 PDF / 批量目录
python -m pipeline ingest 论文.pdf
python -m pipeline ingest-dir ./pdf/

# 查询 (默认 Agentic RAG)
python -m pipeline query --query "MoS2 的晶格常数是多少?"
python -m pipeline query --query "..." --simple --mode vector --top-k 10 --stream
```

---

## 二、后端部分（`pipeline/api/`）

基于 **FastAPI** 的 HTTP 服务，所有接口挂载在 `/api/v1` 前缀下，流式接口使用 SSE。

### 主要能力

- **查询 / 对话**：`POST /query`（单次）、`POST /chat`（多轮）、`POST /chat/stream`（SSE 流式，支持专家模式思考过程实时推送）。
- **知识库管理**：列表 / 新建 / 删除 / 重建（`/collections`），每个知识库对应一个 Milvus 集合 + 本地工作目录，删除连带清理、重建复用解析产物。支持中文库名（自动生成 ASCII slug）。
- **灌入**：上传 PDF 自动灌入（`/ingest/upload`）、全量重灌 / 增量追加 / 仅解析 / 灌入向量化文件，均为异步任务，经 `/tasks/{id}` 轮询进度。
- **专家技能**：列表 / 模版 / 新建编辑 / 删除（`/skills`），用户自定义技能即时生效。
- **运维与日志**：健康检查（`/health`）、集合统计（`/stats`）、文献简介（`/doc_summary`），以及按 session 收集的检索流程日志（`/logs/sessions`，含 SSE 实时流）。

### 认证

Bearer Token 认证，由环境变量 `API_KEYS` 控制；未配置时免认证（本地联调默认）。

> 完整接口规格见 [`docs/后端协议文档.md`](docs/后端协议文档.md)。

### 启动

```bash
# 方式一: 启动器 (自动设置 INFO 日志级别)
CONFIG_PATH=local_api_config.yaml .venv-api/bin/python run_api.py

# 方式二: uvicorn 直接启动
CONFIG_PATH=local_api_config.yaml CORS_ORIGINS="*" \
  .venv-api/bin/python -m uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8080
```

---

## 三、前端部分（`frontend/`）

基于 **React + TypeScript + Vite + TailwindCSS** 的单页管理界面。

### 功能模块

- **智能问答**：快速检索 / 专家模式切换，流式渲染回答与「思考过程」，引用角标可溯源到文献片段，历史对话本地留存。
- **知识库管理**：知识库列表、新建、上传 PDF 灌入、删除、重建，实时任务进度。
- **专家技能**：可视化维护自定义研究技能。
- **系统状态**：各依赖服务（Milvus / LLM / Embedding / Reranker）健康状态。
- **检索日志**：按会话查看检索流程日志，支持实时流式追踪。

开发期通过 Vite 代理 `/api` 到后端（默认 `http://localhost:8080`）。

### 启动

```bash
cd frontend
npm install
npm run dev      # 默认 http://localhost:5173
```

---

## 配置

核心配置见 `pipeline/default_config.yaml`，支持三层覆盖：默认配置 → 用户配置（`--config`）→ 环境变量引用（`${ENV_VAR}`）。

| 配置节 | 关键字段 | 说明 |
|--------|---------|------|
| `parsing` / `mineru` / `uniparser` | `backend`, `authorization`, `api_key` | PDF 解析后端与凭证 |
| `embedding` | `api_base`, `api_key` | Embedding 服务 |
| `milvus` | `uri`, `collection`, `dim` | 向量库连接 |
| `generation` | `api_base`, `api_key`, `model` | 生成 LLM |
| `retrieval.langgraph` | `reranker`, `reflection`, `professional` | Agentic / 专家模式相关 |

> ⚠️ 请勿将真实 API Key 提交到仓库，建议通过环境变量注入。

---

## 主要技术栈

- **算法/后端**：Python、FastAPI、LangGraph、Milvus、Pydantic
- **前端**：React 18、TypeScript、Vite、TailwindCSS
- **解析/模型**：MinerU、UniParser、Embedding / Reranker / LLM（OpenAI 兼容接口）
