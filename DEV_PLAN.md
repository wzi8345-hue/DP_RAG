# DP-RAG 开发计划（源真相 · Source of Truth）

> 与 [`ARCHITECTURE.md`](./ARCHITECTURE.md) 并列的源真相。**每个功能 / issue 的新增排期、每个任务的完成状态都记录在此**。一切开发首要参考本文件与架构文档；每一步推进都要 review 并更新两者。
>
> 状态图例：`☐ 待办` · `◐ 进行中` · `☑ 完成` · `⊘ 取消/暂缓`
>
> 最近更新：2026-06-09

---

## 0. 里程碑总览

| 阶段 | 主题 | 状态 |
|------|------|------|
| M0 | 项目分层 + 源真相文档 | ☑ |
| M1 | 前端脚手架（Vue 栈 + Logto + 布局/主题/i18n） | ☑ |
| M2 | 前端核心页面（问答 / 文献管理 / skill / 设置） | ☑ |
| M3 | 后端 Logto JWT 鉴权 + 用户上下文 | ☑ |
| M4 | 后端 Postgres 消息树 + 对话 CRUD + 分叉重生成 | ☑ |
| M5 | 文献库/skill 归属与可见性（private/org/public） | ☑ |
| M6 | 「断连不停 + 重连续读 + 停止」生成解耦 | ☐ |
| M7 | 多检索源抽象（literature 落地，SQL 预留） | ◐ |
| M8 | 部署（docker-compose + Dockerfile）+ CI/CD + RustFS 对象存储 | ☑ |
| M9 | 基建与约定（uv/ruff · psycopg/pydantic · POST+APIResponse · 备份） | ◐ |

> 说明：M2 前端页面已实现并通过 typecheck/build/test/lint；其中依赖后端的能力（单篇文献过滤 `doc_ids`、按库列文献/删文献、可见性切换、断连重连续读）前端已接好接口，等待对应后端阶段（M4/M5/M6）落地后端点即生效。

---

## 1. 任务清单（按阶段）

### M0 · 项目分层 + 文档
- ☑ #0.1 目录重排：`backend/`（迁移 pipeline/ragas_eval/synthetic_qa_gen/run_api）、删除旧 React `frontend/`、新建 `deploy/`
- ☑ #0.2 编写 `ARCHITECTURE.md`
- ☑ #0.3 编写 `DEV_PLAN.md`
- ☐ #0.4 更新根 `README.md` 反映新结构与启动方式

### M1 · 前端脚手架
- ☑ #1.1 `frontend/` Vite 8 + pnpm + TS 工程（package.json / tsconfig / vite.config）
- ☑ #1.2 UnoCSS（presetWind3 + presetIcons + 主题 token）配置
- ☑ #1.3 vue-i18n（zh-CN / en）骨架
- ☑ #1.4 `@logto/vue` 插件 + AuthGate + `/callback`
- ☑ #1.5 主题：light/dark/system + 自定义主题色（`useTheme`）
- ☑ #1.6 应用骨架：侧栏 + 顶栏 + 路由（类 Vercel 控制台）
- ☑ #1.7 Vitest + Oxlint 配置与最小测试（citations 单测通过）
- ☑ #1.8 API 客户端：`useApi`（注入 access_token）+ 类型

### M2 · 前端核心页面
- ☑ #2.1 问答页：消息树主线渲染（markstream-vue）+ 引用角标
- ☑ #2.2 问答页：发送/停止/编辑历史重生成（分叉）/正在生成禁发
- ◐ #2.3 问答页：SSE（fetch+ReadableStream，带 token）✓；断连续读接口已接（`resumeMessageStream`），等待后端 M6
- ☑ #2.4 问答页：选择单篇文献 / 整库 / 上传文献 / 检索开关 / 专家模式 / 检索源（doc 过滤待后端 M5）
- ☑ #2.5 引用查看面板：有效引用文献角标点击 → 简介 + 命中片段 + 原文 PDF tab 按页跳转
- ☑ #2.6 文献管理页：库 CRUD + 文献上传解析入库 + 文献增删查 + private/org/public 分组 + 复制到个人
- ☑ #2.7 skill 管理页：列表/新建/编辑/删除 + private/org/public 分组 + 复制到个人
- ☑ #2.8 设置页：主题/语言/默认检索参数/检索源/账号登出
- ☑ #2.9 历史对话侧栏：列表/切换/删除/分支切换

### M3 · 后端鉴权
- ☑ #3.1 `pipeline/auth/`：JWKS 缓存 + JWT 校验（iss/aud/exp/scope）
- ☑ #3.2 `AuthContext`（user_id/org_id/scopes）+ `require_auth` 依赖
- ☑ #3.3 替换各 router 的 `verify_api_key` → `require_auth`；运维接口保持免认证
- ☑ #3.4 `AUTH_DISABLED` 本地开发开关
- ☑ #3.5 CORS 调整（rag.hal9k.one / localhost:9527）+ root_path（反代前缀）

> 待验证：需在装好依赖（PyJWT 等）的环境运行，用真实 Logto token 端到端验证。

### M4 · 对话消息树
- ☑ #4.1 数据模型 conversations / messages（pydantic + psycopg DDL，`pipeline/db/`）
- ☑ #4.2 启动建表（lifespan `init_db()` 幂等）+ shell 全量备份（`deploy/backup.sh`）替代迁移框架
- ☑ #4.3 对话 CRUD 路由（列表/读取/visibility，M9.6 统一 POST+APIResponse 待迁移）
- ☑ #4.4 消息追加 / 主线推导 / copy-on-continue（非 owner 继续对话先复制主线）
- ☑ #4.5 chat/stream 落库消息树（保留 SessionStore 作为 pipeline 多轮上下文）

### M5 · 归属与可见性
- ☑ #5.1 kb_collections / documents / user_skills 模型扩展到 `private|org|public`
- ◐ #5.2 文献库按 owner 逻辑隔离（owner + visibility + repo 校验；物理目录仍兼容旧 `uploads/kb_*`，后续可迁 `UPLOAD_DIR/<owner>/`）
- ☑ #5.3 列表/读写鉴权（private(mine) ∪ org(my_org) ∪ public）
- ☑ #5.4 可见性切换接口（collections / skills / conversations）
- ☑ #5.5 对话分享链接（创建/撤销/公开只读解析）+ copy-on-continue 独立副本
- ☑ #5.6 文献库 / skill copy-to-mine：复制为 owner=me 的独立私有资源
- ☑ #5.7 chat/query 检索范围校验：collection 必须可读；professional 自定义 skill 按当前用户可读 allowlist 过滤

### M6 · 生成解耦
- ☐ #6.1 生成移入后台任务 + 按 message_id 缓冲
- ☐ #6.2 `GET /messages/{id}/stream` 重连续读
- ☐ #6.3 `POST /messages/{id}/stop` 取消标志
- ☐ #6.4 前端断连不取消后台；重连恢复

### M7 · 多检索源
- ◐ #7.1 `RetrievalSource` 协议 + `LiteratureSource`（`pipeline/retrieval_sources/` 已落地，待接入 QueryFlow）
- ◐ #7.2 请求体 `sources[]`（前端已发送；后端融合待落地）
- ◐ #7.3 `EnterpriseSqlSource` 接口占位（已建占位类，前端 Beta/灰显）

### M9 · 基建与约定
- ☑ #9.1 后端改用 **uv + ruff**（`backend/pyproject.toml` + `uv.lock`；删除 requirements.txt；ruff 配置）
- ☑ #9.2 DB 改用 **psycopg 3 + pydantic**（删除 SQLAlchemy/Alembic）；lifespan 自动建表
- ☑ #9.3 **shell 全量备份/恢复**（`deploy/backup.sh` / `restore.sh`，`pg_dump | gzip`）
- ☑ #9.4 **APIResponse** 统一封装模型 + `ok/fail` 助手（`pipeline/api/models.py`）
- ☑ #9.5 后端 Dockerfile 改用 uv（`uv sync`）
- ☑ #9.8 pipeline 基建连接支持**环境变量覆盖**（`_ENV_OVERRIDES`）；镜像无需挂载 YAML，compose 注入连接变量
- ☐ #9.6 **API 约定迁移**：现有继承端点统一为 POST + APIResponse（动词式路径），同步更新前端 `api/client.ts`/类型与 `docs/后端协议文档.md`
- ☐ #9.7 项目级 ruff `check --fix` + format 全量过一遍历史 pipeline 代码

> 约定（强制，见 ARCHITECTURE §6.0）：业务接口统一 POST + `APIResponse{code,data,msg}`；SSE 与运维探针（GET /health、/stats）为例外。新增/重做端点（M4/M5）即按此实现。

### M8 · 部署 + CI
- ☑ #8.1 `backend/Dockerfile`
- ☑ #8.2 `frontend/Dockerfile`（多阶段：build → nginx）+ `nginx.conf`
- ☑ #8.3 `deploy/docker-compose.yaml`（backend + postgres）+ `.env.example`
- ☑ #8.6 RustFS/S3 compatible 对象存储服务与 env 配置（PDF 原文 + 解析产物 + 预签名 URL）
- ☑ #8.4 CI：变更检测（paths）+ 构建推送私有仓库镜像（DP_USERNAME/DP_PASSWORD）
- ☑ #8.5 CI：前端变更构建并部署 GitHub Pages（CNAME rag.hal9k.one）

> 待验证：CI 需推送到 GitHub 后实际跑通；GitHub Pages 需在仓库 Settings 启用并指向 Actions。

---

## 2. 变更日志（Changelog）

> 每次推进在此追加一行（日期 · 内容 · 影响的 issue）。

- 2026-06-08 · 初始化：完成 M0 目录重排与两份源真相文档（#0.1 #0.2 #0.3）。
- 2026-06-08 · M1/M2：Vue 前端全量重写完成（脚手架 + 问答/文献/skill/设置页 + Logto + 主题 + i18n + 消息树/分叉 + 引用面板）；typecheck/build/test/lint 全绿。
- 2026-06-08 · M3：后端 Logto JWT 本地校验（`pipeline/auth/`）+ `require_auth` 替换所有业务路由的 API Key 鉴权；CORS/root_path/AUTH_DISABLED 就绪。
- 2026-06-08 · M4/M5/M7：数据模型（conversations/messages/kb_collections/documents/user_skills）与多检索源抽象（literature/enterprise_sql）脚手架落地；路由集成为下一阶段。
- 2026-06-08 · M8：前后端 Dockerfile + `deploy/`（compose + .env.example）+ 三个 GitHub Actions（前后端镜像构建推送私有仓库、前端部署 Pages）。
- 2026-06-08 · M9 基建/约定更新：后端切到 **uv + ruff**（`pyproject.toml` + `uv.lock`，删 requirements）；DB 改 **psycopg 3 + pydantic**（删 SQLAlchemy/Alembic），lifespan 自动建表；新增 **shell 全量备份/恢复**；新增 **APIResponse** 统一封装；确立「业务接口统一 POST + APIResponse」约定（见 ARCHITECTURE §6.0）。新代码 ruff 全过、后端 Dockerfile 改 uv。
- 2026-06-09 · M9 基建：pipeline 基建连接支持**环境变量覆盖**（`config.py` 新增 `_ENV_OVERRIDES`，优先级 default<file<env<runtime），容器化部署无需挂载 YAML；compose 注入 `EMBEDDING_/MILVUS_/LLM_/RERANKER_/REFLECTION_/PARSER_*` 等变量，后端 Dockerfile 默认 `CONFIG_PATH=""`，`local_api_config.yaml` 标注为本机联调用。文档（ARCHITECTURE §10、.env.example）同步。
- 2026-06-09 · UI 十五轮（Popover 暗色对比 + 英文溢出）：新增 `--popover-bg` / `--popover-border`，暗色浮层背景提升到独立层级并增强 `shadow-pop`；Naive `popoverColor` 同步使用浮层色。账号菜单一级项改为固定 grid 列（icon / label / value / chevron），英文长 label/value 使用省略号；二级菜单选项 label 同样使用固定列与省略号，避免英文状态下文本重叠。UI_STYLE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 十四轮（账号邮箱优先）：Logto 前端默认 scopes 从仅 `all:data` 调整为 `all:data,email,profile`，CI/Pages 构建环境同步，确保 `fetchUserInfo()` 可返回邮箱资料；账号菜单与设置页账号展示不再使用 `sub` 兜底，避免直接暴露 Logto user_id。ARCHITECTURE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 十三轮（账号菜单二级偏好菜单）：`UserMenuPopover` 中语言偏好/主题偏好不再直接内联展示分段按钮，改为一级菜单项 + 右侧二级 `NPopover`；二级菜单选项左侧固定预留 check icon 列，当前选中项显示勾号，未选项留空，保证 label 对齐。UI_STYLE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 十二轮（Popover 内直接设置偏好）：`UserMenuPopover` 中语言偏好改为直接调用 `setLocale()` 切换中/英，主题偏好改为直接写入 `useSettings().theme` 切换浅色/深色/跟随系统；只有「通用设置」继续跳转到 `/settings#general`。`AppLayout` 移除语言/主题标签计算，仅负责账号信息与路由事件。UI_STYLE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 十一轮（侧栏账号 Popover）：参考 ChatGPT 网页，将一级菜单底部改为只显示账号入口，主展示优先使用邮箱；移除底部直出的设置/主题切换/服务状态。新增 `UserMenuPopover`，点击账号后用 `NPopover` 分组展示用户信息、语言偏好、主题偏好、通用设置与退出登录；设置页补 `#language/#theme/#general` 锚点。UI_STYLE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 十轮（固定外框 + 触控板手势保护）：`html/body/#app/app-shell/app-workspace/workspace-card` 统一禁止外层滚动与 overscroll，内部滚动容器使用 `overscroll-behavior-y: contain`，避免非滚动区域滚动时整体页面被拖动；新增 `usePreventNavigationSwipe()`，拦截不可由内部横向滚动容器消费的触控板横向 wheel，降低误触浏览器前进/后退。UI_STYLE 同步。typecheck/build/lint 全绿。
- 2026-06-09 · UI 九轮（细节修正）：一级菜单和二级菜单之间 `app-shell` gap 调整为 8px；二级菜单↔内容区 `WorkspaceSplit` 热区调整为 12px，默认不显示分割线，仅 hover 时显示 1px 深色细线。设置页主题切换改为 `NTabs type="segment"` 以明确当前状态，主题色取色器扩宽并限制 hex 模式，`API base` label 固定宽不换行且输入框缩小；智能问答二级头部改为垂直布局（新对话按钮一行，搜索按钮放入输入框 suffix），搜索区和历史/检索结果列表之间改为明确 1px `.menu-section-divider` 分割线，列表改用 `NVirtualList`。UI_STYLE 同步。
- 2026-06-09 · UI 八轮（固定一级菜单 + NButton 统一）：删除一级菜单与业务区之间的外层 `NSplit`，一级菜单固定 168px；二级菜单↔内容区继续使用 `WorkspaceSplit(NSplit)`。智能问答二级菜单头部改为 flex：对话搜索框 + `NButton tertiary` 新建对话。业务代码中原生 `<button>` / `btn-*` / `tbtn` 全部替换为 Naive UI `NButton`，并在 `useNaiveTheme` 配置 Button 主题（无边框/无阴影、primary/tertiary/quaternary/error 与 token 对齐）。UI_STYLE 同步。
- 2026-06-09 · UI 七轮（飞书式三栏布局）：按反馈将一级菜单也纳入 `WorkspaceSplit(NSplit)`，一级菜单处于最外层浅色 `--chrome-bg` 背景；右侧业务区留白。二级菜单与内容区统一包成 `workspace-card` 圆角半透明面板，一级菜单↔工作区、二级菜单↔内容区均由 `NSplit` 14px 统一热区 + 1px 主题线分割；去掉一级/二级硬边侧栏和手写 `border-r`。SourcesPanel 同步为圆角面板；新建对话按钮改为主按钮，避免浅灰底过重。UI_STYLE 同步。
- 2026-06-09 · UI 六轮（NSplit 工作区分割）：排查到分割线不可见不是单纯 UnoCSS 未生成，而是 1px `border-r` 在浅色白底下对比弱，且 Chat/Library/Skills 分别硬编码 `w-60/w-72` 导致二级栏宽度不一致。新增 `WorkspaceSplit`（Naive UI `NSplit` 包装，默认 264px、范围 220–360px、7px 拖拽热区 + 1px 主题线），Chat/Library/Skills 统一改用它；`useNaiveTheme` 配置 `Split.resizableTriggerColor*`，`theme.css` 增 `.workspace-split-trigger` 适配 light/dark。UI_STYLE 同步。
- 2026-06-09 · UI 五轮（暗色状态与分割线）：新增 `--active/--active-hover`，暗色 `--hover`/分割线加深；新增 `--accent-visible`，暗色下将过暗主题色抬亮，避免黑色 accent 导致按钮/状态不可见。一级/二级/右侧容器统一 `bg-surface`，用 `border-r/l` 和 `border-b` 细分割线表达结构；对话、文献库、skill 列表、设置/Composer 分段按钮、来源面板 tab 均改用 `bg-active` 选中态。UI_STYLE 同步。
- 2026-06-09 · UI 四轮（根因：缺少 reset 导致原生 button chrome）：定位"按钮仍有边框"真因是 **UnoCSS 默认不含完整 preflight**，原生 `<button>` 暴露浏览器默认 1px 边框+灰底（纯色按钮被填充盖住，ghost/分段按钮露出）。在 `theme.css` 对 `button` 做中性化 reset（去 border/background/appearance/下划线）。同时按钮间距加大：控制行/动作组 `gap-2`、图标组 `gap-1.5`。Settings 检索源原生 checkbox → `NCheckbox`。UI_STYLE 增"原生 chrome 中性化 + 按钮 gap"规则。typecheck/build/lint 全绿。
- 2026-06-09 · UI 三轮（引入 Naive UI 解决原生控件边框）：定位到「按钮仍有边框 / 输入框四边边框色不一致」实为**原生 `<select>`/`<input>` 浏览器 chrome**。按要求引入 **Naive UI** 承载复杂控件：`App.vue` 加 `NConfigProvider` + `useNaiveTheme()`（主题随 `--accent` + light/dark，输入框四边同色细边框）；Settings(mode→NSelect、top_k→NInputNumber、apiBase→NInput、主题色→NColorPicker)、Composer(文献库→NSelect、选择文献→NPopover+NCheckbox)、Library(库名→NInput、文献表→NDataTable)、Skills(表单→NInput/NInputNumber)。`useTheme` 导出 `useIsDark`。`index.html` 加 `naive-ui-style` 锚点。NModal/NSplit/NScrollbar(全局)/NVirtualList 按需增量采用。UI_STYLE.md 增「组件库」章。typecheck/build/lint 全绿。
- 2026-06-09 · UI 二轮（按反馈逐组件落地）：明确「下滑线 = 按钮文字下划线」并全局兜底 `a,button{text-decoration:none}`（正文链接 hover 下划线保留）；问答控制行/文献库下拉由描边方块改 Vercel 式幽灵工具按钮 `tbtn`/`tbtn-on`；助手回答去盒子（直接渲染于背景），来源/引用条目、技能提示、编辑框去边框盒子改浅填充；横向分割线降为 `border-softer`（极细），仅保留窗框级分割；整体留白取「适中」（chat/library/skills 头部与内容内边距上调、消息间距加大）。typecheck/build/lint 全绿。
- 2026-06-09 · UI 风格固化与优化：新增根目录 `UI_STYLE.md`（Vercel 风格源真相）。按钮去 border/shadow（主=纯色主题色 `btn-primary`，次=主题色透明填充 `btn-secondary`，`btn-outline` 已淘汰）；`chip` 去边框；新增 `--accent-soft/-hover`（随主题色派生）与 `bg-accent-soft*`、`shadow-pop`；滚动条改细 + `scrollbar-gutter: stable`（不抖动）；大圆角降到 8px、移除浮层 `shadow-lg`；分段控件改填充轨道；SettingsPage 由多卡片改为单面板 + 分割线。typecheck/build/lint 全绿。
- 2026-06-09 · 修复登录死循环（M1 #1.4）：`App.vue` 原先用 `@logto/vue` 的 `isLoading` 作为渲染门控；而 `isLoading` 在每次 `getAccessToken/fetchUserInfo` 时都会翻 true，导致 AppLayout 挂载→触发请求→isLoading=true→卸载→resolve 后重新挂载→再请求…的死循环，持续打后端。改为用 `isLoading` 锁存「首次鉴权完成」一次（`initialAuthChecked`），之后仅按 `isAuthenticated` 门控，AppLayout 不再被反复挂载/卸载。
- 2026-06-09 · 引用跳转 + 对象存储 + 三级权限：新增 RustFS/S3 client（`boto3`）与 compose 服务/env；上传入库将 PDF 与解析产物写入对象存储并落 `documents.pdf_object_key/artifact_prefix`；新增 `POST /documents/pdf-url` 预签名 URL。前端引入 `@embedpdf/vue-pdf-viewer`，SourcesPanel 增「原文」tab，点击引用角标按 `hit.page_start` 跳页。完成 `Visibility=private|org|public`、`api/authz.py`、`db/repo.py`、collections/skills/conversations/chat/query 权限接线；新增分享链接、copy-on-continue、文献库/skill copy-to-mine 与前端 public→org→mine 分组。验证：frontend typecheck 通过；后端变更文件 `py_compile` 通过（全量 ruff 仍有历史 typing/ruff 债务，见 #9.7）。
- 下一步：M9.6 现有端点统一迁移到 POST+APIResponse（含前端 client 与协议文档）→ M6 生成解耦（断连不停/重连续读/停止）→ M7 多源融合。

---

## 3. 待澄清 / 风险

- 组织（org）信息如何随 Logto token 下发：需确认 access_token 是否含 `organizations` / `organization_id` claim；否则 org-public 需通过 Logto Management API 或固定单组织实现。【M5 风险】
- `funmg.dp.tech/sci-loop-api` 反代是否会重写路径前缀（影响前端 `VITE_API_BASE` 与后端 root_path）。【M8】
- Milvus 多用户隔离：当前按集合命名 + owner 校验；如需更强隔离可评估 partition/db。【M5】
- 文献库 copy-to-mine 当前会复制 DB metadata、本地解析产物并提交重建任务；对象存储中的 PDF key 会复用源 key。若后续实现源对象硬删除，需要补对象存储前缀级复制或引用计数。【M5】
- 生成缓冲当前用进程内内存；多副本部署需换 Redis/PG LISTEN。【M6 风险】
