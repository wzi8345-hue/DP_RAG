# DP-RAG · 科学知识问答平台

面向科研文献的端到端 **Agentic RAG** 系统，支持 **多用户**（Logto 登录鉴权、数据按用户隔离、可选组织内共享），提供文献管理、专家技能管理与可溯源的智能问答。

> 📐 架构与计划是项目的两份**源真相文档**，一切开发首要参考、每步推进同步更新：
> - [`ARCHITECTURE.md`](./ARCHITECTURE.md) — 前后端架构、数据模型、鉴权、部署
> - [`DEV_PLAN.md`](./DEV_PLAN.md) — 开发计划、排期与每个任务的完成状态

```
浏览器 ── Logto 登录 ──► rag.hal9k.one (前端, GitHub Pages)
   │ access_token (JWT, audience = sci-loop-api)
   └──► funmg.dp.tech/sci-loop-api (后端 FastAPI) ──► Postgres / Milvus / LLM
```

---

## 仓库结构

```text
DP_RAG/
├── ARCHITECTURE.md / DEV_PLAN.md   # 源真相文档
├── backend/                        # Python 后端（FastAPI + RAG pipeline）
│   ├── pipeline/
│   │   ├── api/                    # FastAPI 路由 / 会话 / 任务 / 日志
│   │   ├── auth/                   # Logto JWT 本地校验 + 用户上下文
│   │   ├── db/                     # psycopg 3 + pydantic（对话消息树 / 归属 / 可见性，无 ORM）
│   │   ├── retrieval_sources/      # 多检索源抽象（literature / enterprise_sql 预留）
│   │   ├── clients/ processors/ retrieval/ routing/ steps/ flows/  # RAG 流水线
│   │   └── skills/                 # 内置专家技能
│   ├── pyproject.toml  uv.lock  Dockerfile  run_api.py  local_api_config.yaml
│   ├── ragas_eval/  synthetic_qa_gen/
├── frontend/                       # Vue 3 前端（Vite 8 / pnpm / UnoCSS …）
└── deploy/                         # docker-compose.yaml + .env.example
```

---

## 快速开始

### 后端

```bash
cd backend
uv sync                       # 依赖与虚拟环境（读取 pyproject.toml / uv.lock）

# 本地联调可跳过鉴权（仅本地！）；DATABASE_URL 未配置时跳过建表
AUTH_DISABLED=1 CONFIG_PATH=local_api_config.yaml uv run python run_api.py   # :8080
```

> 依赖与 lint 用 **uv + ruff**：`uv sync`（装依赖）、`uv run ruff check .`（lint）、`uv run ruff format .`（格式化）。数据库直接用 **psycopg 3**，表结构在服务启动（lifespan）自动检查/初始化；备份/恢复见 `deploy/backup.sh` / `deploy/restore.sh`。

鉴权（生产）：后端用 JWKS 本地校验 Logto JWT（issuer `auth.dplink.cc/oidc`，audience `funmg.dp.tech/sci-loop-api`，scope `all:data`）。配置见环境变量（`LOGTO_*`、`DATABASE_URL`、`CORS_ORIGINS`、`API_ROOT_PATH`），详见 [`deploy/.env.example`](./deploy/.env.example)。

### 前端

```bash
cd frontend
pnpm install
pnpm dev          # http://localhost:9527 （与 Logto 重定向 URI 一致）
```

本地默认把 `/api` 代理到 `http://localhost:8080`。详见 [`frontend/README.md`](./frontend/README.md)。

### 后端 API

所有业务接口挂载在 `/api/v1` 前缀下，统一返回 `APIResponse{code,data,msg}`；流式问答与日志走 SSE（`text/event-stream`），运维探针 `GET /health`、`GET /stats` 为例外。请求需携带 Logto 签发的 JWT：`Authorization: Bearer <access_token>`。

接口契约以服务运行时的 OpenAPI 为准（源真相），启动后端后访问：

- Swagger UI：`/docs`
- ReDoc：`/redoc`
- OpenAPI JSON：`/openapi.json`

端点分组与约定详见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §10。

### 部署（docker-compose）

```bash
cd deploy
cp .env.example .env      # 编辑 Logto / DATABASE_URL / Milvus 配置
docker compose --env-file .env up -d
```

---

## 核心能力

### RAG 流水线（`backend/pipeline/`）

- **灌入**：`PDF → parse(MinerU/UniParser) → chunk → embed → store(Milvus)`，按知识库隔离落盘，支持重建 / 增量追加。
- **查询**：三档检索逐级增强——基础 `hybrid/vector/metadata`、**Agentic RAG**（LangGraph 路由 + 多路径检索 + 重排 + 反思）、**专家模式**（多轮递进研究 + 综述综合 + 技能定制）。

### 多用户与鉴权

- Logto OIDC 登录；后端本地校验 JWT；文献库 / 对话 / skill **按用户隔离**，owner 可设为「组织内部可读」共享（`visibility = private | org`）。

### 对话消息树

- 对话以 **消息树** 记录多轮：每条 message 记录 `parent_id`；编辑历史输入重新生成 → 从该处**分叉**；`active_leaf_message_id` 沿父指针回溯并反转 = 当前主线。前端已实现该模型，后端 Postgres 落库逐步接管。

### 前端（`frontend/`）

Vue 3 · `<script setup>` · TS · Vite 8 · pnpm · UnoCSS（presetIcons）· vue-i18n · `@logto/vue` · markstream-vue · VueUse · Vitest · Oxlint。风格类 Vercel 控制台（轻量、平铺、低视觉噪音，light/dark/system + 自定义主题色）。

页面：智能问答（停止 / 编辑重生成分叉 / 断连后台续跑 / 引用查看）、文献管理（库与文献增删改查、上传解析、可见性）、专家技能管理、设置。

---

## CI/CD

GitHub Actions（`.github/workflows/`）：

- `backend/**` 变化 → 构建并推送后端镜像到私有仓库；`frontend/**` 变化 → 构建前端镜像 + 部署 GitHub Pages（`rag.hal9k.one`）。
- 镜像命名：`dp-harbor-registry.cn-zhangjiakou.cr.aliyuncs.com/dplc/qsar:sci-loop_{backend|frontend}_{yymmdd}_{sha4}`。
- 私有仓库登录用仓库 secrets `DP_USERNAME` / `DP_PASSWORD`。

---

## 文档

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)、[`DEV_PLAN.md`](./DEV_PLAN.md) — 源真相文档（架构、数据模型、鉴权、部署、API 约定与端点）
- [`deploy/.env.example`](./deploy/.env.example) — 部署环境变量说明
- 后端 API 契约：运行后端后看 `/docs`、`/redoc`、`/openapi.json`（OpenAPI 自动生成，始终与代码一致）

> ⚠️ 请勿将真实 API Key / 密码提交到仓库，统一通过环境变量 / secrets 注入。
