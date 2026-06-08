# DP-RAG 前端

基于 `pipeline/api`（FastAPI）的 Web 前端：科研文献智能问答、知识库管理、系统状态监控。

技术栈：Vite + React 18 + TypeScript + Tailwind CSS v4。

## 功能

- **智能问答**：多轮对话、SSE 流式输出、Agentic/检索模式切换、引用来源溯源面板。
- **知识库**：PDF 上传、向量库重灌/增量追加、PDF 解析、向量文件灌入，异步任务进度轮询。
- **系统状态**：健康检查（Milvus / LLM / Embedding / Reranker / Reflection）、Milvus 集合统计。
- **设置**：后端地址、API Key、检索模式、Top-K、流式开关，保存在浏览器本地。

## 对接的后端接口

| 方法 | 路径 |
|---|---|
| POST | `/api/v1/query` |
| POST | `/api/v1/chat` |
| POST | `/api/v1/chat/stream` (SSE) |
| POST/DELETE | `/api/v1/sessions[/{id}]` |
| POST | `/api/v1/files/upload` |
| POST | `/api/v1/ingest/{rebuild,append,parse,load-vec}` |
| GET | `/api/v1/tasks/{id}` |
| GET | `/api/v1/stats`, `/api/v1/health` |

## 开发

```bash
cd frontend
npm install
npm run dev
```

默认 `http://localhost:5173`。开发服务器把 `/api` 代理到后端（默认 `http://localhost:8080`）。
后端在别处时：

```bash
VITE_API_TARGET=http://192.168.1.10:8080 npm run dev
```

或在「设置」里直接填写后端 Base URL（留空则走开发代理）。

## SSH 隧道（连接远程 GPU / Milvus 服务）

后端通过 SSH 本地端口转发访问远程服务器上的服务（LLM 8000 / reranker 8001 / embedding 8002 / Milvus 19530 / 3000）。

### 一次性：把 key 密码存入 Keychain（key 带密码时必做）

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # 手动输入一次密码，存入钥匙串
```

完成后，下面两种方式都能**无人值守自动连接**。

### 方式一：启动前端时自动连接

`npm run dev` 会先执行 `predev` 钩子（`scripts/tunnel.sh`）自动拉起隧道（已连接则跳过）。
也可单独运行：

```bash
npm run tunnel
```

### 方式二：永久后台服务（开机自启 + 断线自动重连，推荐）

基于 macOS launchd + autossh：

```bash
npm run tunnel:install     # 安装并启动（日志: /tmp/dprag-tunnel.log）
npm run tunnel:uninstall   # 停止并卸载
```

> 参数（服务器 IP、端口、转发列表）写在 `scripts/tunnel.sh` 与 `scripts/com.dprag.tunnel.plist`，需要变更时直接编辑。

## 构建

```bash
npm run build      # 产物输出到 dist/
npm run preview
```

生产环境部署时，在设置里填写后端绝对地址，或用反向代理把 `/api` 转发到 FastAPI。
