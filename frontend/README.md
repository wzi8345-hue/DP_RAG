# DP-RAG 前端

科学知识问答平台前端。Vue 3 · Vite 8 · pnpm · UnoCSS · vue-i18n · `@logto/vue` · markstream-vue · VueUse · Vitest · Oxlint。

风格参考 [`../ARCHITECTURE.md`](../ARCHITECTURE.md) 与 [`../DEV_PLAN.md`](../DEV_PLAN.md)（源真相文档）。

## 开发

```bash
pnpm install
cp .env.example .env.local   # 按需修改；本地留空 VITE_API_BASE 走 dev proxy
pnpm dev                     # http://localhost:9527 （与 Logto 重定向 URI 一致）
```

后端默认代理到 `http://localhost:8080`，可用 `VITE_API_TARGET` 覆盖。

## 脚本

| 命令 | 说明 |
|------|------|
| `pnpm dev` | 开发服务器（端口 9527） |
| `pnpm build` | 类型检查 + 生产构建（产出 `dist/`，含 `404.html` SPA 回退与 `CNAME`） |
| `pnpm preview` | 预览构建产物 |
| `pnpm test` | Vitest 单测 |
| `pnpm lint` | Oxlint |
| `pnpm typecheck` | vue-tsc 类型检查 |

## 目录

```
src/
├── api/         # 类型、HTTP 客户端、SSE
├── auth/        # Logto 配置
├── components/  # 复用组件（消息气泡、来源面板、composer…）
├── composables/ # useApi / useChat / useTheme / useSettings
├── i18n/        # zh-CN / en
├── pages/       # chat / library / skills / settings / callback
├── stores/      # 对话消息树（pinia）
├── styles/      # 主题 token
└── utils/       # 引用解析、LaTeX 兜底
```

## 鉴权

`@logto/vue` 取 access_token（`getAccessToken(API_RESOURCE)`），所有后端请求与 SSE 都带 `Authorization: Bearer <jwt>`。SSE 用 `fetch + ReadableStream`（不用 EventSource，以便携带 token）。

## 部署

CI 在 `frontend/**` 变化时构建并部署到 GitHub Pages（域名 `rag.hal9k.one`，见 `public/CNAME`）。`VITE_API_BASE` 在 CI 注入为 `https://funmg.dp.tech/sci-loop-api`。
