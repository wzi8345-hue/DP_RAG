# DP-RAG 前端 UI 风格规范（源真相）

> 参考 **Vercel 控制台**的干净、低视觉噪音风格。本文件固化前端视觉与交互偏好，指导后续所有 UI 设计与组件开发。改动 UI 前先读本文件；新增/调整规范时同步更新这里。
>
> 配套源真相：[`ARCHITECTURE.md`](./ARCHITECTURE.md) §7（前端架构）、`frontend/uno.config.ts`（shortcuts/rules）、`frontend/src/styles/theme.css`（token）。

---

## 1. 设计原则

1. **低噪音、平铺**：少边框、少阴影、少填充块；用**窗框级极细分割线 + 留白**组织布局，而非"卡片套卡片"。
   - 留白密度取**适中**（airy 与 compact 之间）：信息密度仍高，但比早期更透气。
2. **内容优先**：界面服务于内容与操作，不做装饰性 hero / 营销区 / 大图标。
3. **一致性来自 token**：所有颜色走 CSS 变量；切换 light/dark/system 与自定义主题色只改变量，不改组件类名。
4. **紧凑清晰**：突出主操作按钮；信息密度高但留白克制。
5. **稳定布局**：滚动条、加载态、内容变化都不应引起布局宽度/位置抖动。

---

## 2. 颜色 token（`theme.css`）

语义变量（组件只用语义类，不写死颜色）：

| 变量 | 含义 | light | dark |
|------|------|-------|------|
| `--bg` | 页面背景 | 纯白 `#ffffff` | 近黑 `#0a0a0b` |
| `--chrome-bg` | 飞书式外层工作台背景 | `#f3f6fb` | `#0f1014` |
| `--surface` | 容器/面板背景 | 白 | `#131316` |
| `--surface-2` | 次级背景（输入回填、代码块、chip） | `#f7f8fa` | `#1a1a1e` |
| `--hover` | hover 态背景 | `#f1f3f5` | `#25252b` |
| `--active` | tab / 列表选中态背景 | `#eceff3` | `#303039` |
| `--active-hover` | 选中项 hover 背景 | `#e6eaf0` | `#383844` |
| `--border` | 主分割线/边框 | `#e4e7eb` | `#303038` |
| `--border-subtle` | 更轻的分割线 | `#eef0f2` | `#25252c` |
| `--text` | 主文字/标题（深色） | `#18181b` | `#f4f4f5` |
| `--text-muted` | 次要描述（稍浅） | `#6b7280` | `#a1a1aa` |
| `--text-faint` | 占位/弱提示（最浅） | `#9ca3af` | `#6b7280` |
| `--accent` | 主题色（可自定义） | `#2563eb` | 同 |
| `--accent-visible` | 暗色模式下抬亮后的可见主题色 | 同 accent | `accent` 向白色混合 |
| `--accent-soft` | 主题色透明填充（次要按钮/角标底） | 由 visible accent 派生 | 同 |
| `--accent-soft-hover` | 主题色透明 hover | 由 visible accent 派生 | 同 |

- **背景**：支持暗黑与明亮白色，跟随系统（`system`）或手动切换；`color-scheme` 同步设置。应用最外层使用 `--chrome-bg`，业务内容使用 `--surface`/半透明 surface 面板。
- **主题色切换**：`--accent` 改变时，`--accent-visible` / `--accent-soft*`（`color-mix` 派生）自动跟随。暗色下如果用户选择黑色/深色 accent，`--accent-visible` 会向白色抬亮，保证按钮和状态可见。

对应 UnoCSS 语义类：`bg-app / bg-surface / bg-surface-2 / bg-hover / bg-active / bg-active-hover / bg-accent / bg-accent-soft`、`text-base / text-muted / text-faint / text-accent`、`border-soft / border-softer`。

---

## 3. 文字层级

- **主标题 / 主文字**：`text-base`（深色，`--text`），字重 600 左右。
- **次要描述 / 辅助说明**：`text-muted`（稍浅）。
- **占位、弱提示、计数**：`text-faint`（最浅）。
- 字号紧凑：正文 14px，说明 12px，标题 16–18px；行高克制。
- 不滥用粗体与色彩；强调靠层级与留白，不靠颜色堆叠。

---

## 4. 按钮（重要）

**统一规则：按钮使用 Naive UI `NButton`，不设 `border`，不设 `shadow`。** 形态靠填充与 hover 反馈，便于主题色切换。

| NButton 用法 | 用途 | 样式 |
|----|------|------|
| `type="primary"` | 主功能（发送、保存、登录、上传…） | 纯色主题色填充 + 白字，hover 轻微提亮 |
| `tertiary` | 次要功能（新建对话、重建、可见性、编辑…） | 静止态低噪音，hover 显示中性底 |
| `quaternary` / `text` | 三级/低权重（取消、关闭、工具按钮、内联操作） | 静止态无填充，hover 显示中性底或主题色文字 |
| `type="error"` | 危险操作（删除、停止） | 红色语义填充或文字，按具体风险选择 |
| `circle + quaternary` | 图标按钮 | 方形/圆形低噪音按钮，hover `bg-hover` |

- **工具栏/控制按钮**统一用 `NButton` 的 `tertiary` / `quaternary` / `text`，不要再写原生 `<button>` 或项目自定义 `btn-*` / `tbtn`。
- **分段按钮 / Tab / 列表选中态**用 `bg-active text-base`，hover 用 `bg-active-hover`；不要用 `bg-hover` 同时承担 hover 与 active，否则暗色模式下状态对比不足。非主操作的 tab 不使用 `bg-accent`，避免黑色 accent 在暗色下不可见。
- 圆角 6px；尺寸紧凑（`px-3 py-1.5`，图标按钮 `h-8 w-8`）。
- 禁用态 `opacity-50 + not-allowed`。
- 图标用 UnoCSS presetIcons（`i-lucide-*`，icon-in-css）；不明显的图标/按钮必须带 `title`/tooltip。
- ❌ 按钮三不要：**不要 border、不要 shadow、不要文字下划线**。
- **原生 chrome 已全局中性化**：`theme.css` 对 `button` 做了 reset（去 `border`/`background`/原生 `appearance`/下划线），主要用于兜底和 Naive 内部 button 继承环境。业务代码新增按钮必须使用 `NButton`。
- **多个按钮之间留 gap**：同组按钮用 `gap-2`（图标按钮组可 `gap-1.5`）；分段控件内部例外（同一控件，`gap-1` + `bg-surface-2` 轨道）。
- ❌ 不要 `btn-outline`、原生 `<button>`、自定义 `btn-*` / `tbtn`（已淘汰）。

---

## 5. 容器、分割与圆角

- **一级菜单在最外层且固定宽度**：`AppLayout` 使用飞书式浅色 `--chrome-bg` 底，一级菜单固定 `168px`，直接浮在外层底色上；一级菜单和业务工作区之间不用 `NSplit`。
- **一级菜单和二级菜单之间固定 8px 间距**：一级菜单内部列表项已有留白与 hover 区分，外层 `app-shell` 的横向 gap 保持 `8px`。
- **二级菜单↔内容区统一使用 `WorkspaceSplit`（Naive UI `NSplit` 包装）**：二级栏默认 `264px`（范围 `220px–360px`），`resize-trigger-size=12` 作为统一拖拽热区和卡片间留白；默认不显示分割线，hover 时显示居中的 1px 深色细线。不要在各路由里再写 `w-60/w-72` 或手动 `border-r`。
- **二级菜单与内容容器使用 `workspace-card` 圆角面板**：圆角 10px，半透明 `surface` 背景，放在 `--chrome-bg` 上；结构靠 `NSplit` 分割和外层留白表达，不靠整块侧栏背景。
- **一级菜单底部只显示账号入口**：参考 ChatGPT 网页，底部不直出服务状态、设置、主题切换等杂项；只保留 `UserMenuPopover` 账号组件，主展示优先使用邮箱。点击后在 `NPopover` 中展示「用户信息｜语言偏好 / 主题偏好 / 通用设置｜退出登录」分组菜单。语言和主题在一级菜单中显示为带右箭头的菜单项，点击后继续弹出二级菜单；二级菜单每一行左侧固定预留 check icon 空间，当前状态显示勾号，未选项保留空位，保证 label 起始列对齐。英文长文本必须使用固定列宽 + 省略号，不能挤压状态文本或 chevron。只有「通用设置」跳转到设置页。
- **智能问答二级菜单头部**垂直布局：第一行 `NButton tertiary` 新建对话（高度用 `size="medium"`，图标+文字从左到右排列）；第二行 `NInput` 搜索框，搜索按钮放入 input 的 `suffix` 槽中。
- 搜索区和历史对话/检索结果列表之间使用 `.menu-section-divider` 分割（明确 1px 高度，颜色走 `--border`，避免 0 高度 border 在浅色卡片中不可见）；历史对话/检索结果列表使用 `NVirtualList` 渲染。
- `WorkspaceSplit` 的分割线样式由 `theme.css` 的 `.workspace-split-trigger` 控制；默认透明，hover 使用 `--text-faint` 显示 1px 细线。
- **区块内部不要再拉横向分割线堆叠**；用留白 + 浅色背景（`bg-surface-2`）/ hover 区分条目，而不是给每条加边框做成盒子。
- **避免卡片套卡片**：一个区域最多一层容器；不要在 card 里再放 card；列表项不要每条都套边框盒子。
- **圆角**：普通控件 6–8px；工作区面板 `workspace-card` 10px。❌ 不用大圆角（>12px）、不用 `rounded-2xl/full`（状态点、头像、色板除外）。
- **填充**：工作台外层用 `--chrome-bg`；工作区圆角面板用半透明 `surface`；次级回填（代码块、chip、非布局型提示）可用 `bg-surface-2`。❌ 不用 Material 那种大面积纯色品牌填充，也不要把侧栏整块染成次级背景。
- `card`（`bg-surface border border-soft rounded-[8px]`）仅用于**确需独立成块**的场景（登录卡、弹出层），不滥用。
- 弹出层（下拉/popover）使用 `--popover-bg` / `--popover-border` / `shadow-pop`，暗色模式下必须明显高于页面背景和卡片背景，避免和工作区混在一起；按钮与普通容器不用阴影。

---

## 6. 滚动条（重要）

- 可滚动区域**显示滚动条**，但**细、低存在感、不引起布局宽度抖动**。
- 实现：
  - 细滚动条（`scrollbar-width: thin` / WebKit 8px），轨道透明，滑块 `--scrollbar-thumb`（半透明，hover 加深）。
  - 纵向滚动容器统一 `scrollbar-gutter: stable`：滚动条出现/消失时**预留稳定 gutter，宽度不抖动**。
  - macOS 等支持悬浮（overlay）滚动条的平台由系统自动隐藏，不额外占用观感空间。
- ❌ 不要强制 10px+ 的常驻粗滚动条挤占内容、导致宽度跳动。

---

## 7. 布局

- 用 **flex / grid + 局部 overflow** 承载工作区；避免浏览器**页面级滚动**承载复杂界面。
- 典型三栏：主导航（最外层浅底、固定宽度）｜ 二级列表/会话（圆角面板）｜ 内容区（圆角面板）；二级列表和内容区之间用 `WorkspaceSplit(NSplit)` 分割，不用手写 `border-r`。
- 列表项 hover 用 `bg-hover`；选中态用 `bg-active text-base`，选中项 hover 用 `bg-active-hover`（不用边框/阴影框住）。
- 表格用细分割线（`border-softer` 行分隔），不加竖线网格。

---

## 8. 反例（不要这样做）

- ❌ 按钮带边框或阴影；❌ 主按钮用浅色描边而非纯色填充。
- ❌ Material 风：卡片套卡片、大圆角、大阴影、厚渐变、大面积品牌色块。
- ❌ 写死颜色（`bg-blue-600` / `text-gray-500`）——一律走语义 token。
- ❌ marketing hero、装饰性插画、超大图标标题区。
- ❌ 常驻粗滚动条 / 滚动条导致内容宽度抖动。

---

## 9. 组件库（Naive UI）

技术较复杂的基础组件**统一用 [Naive UI](https://www.naiveui.com/)**，不用原生 HTML 控件（避免原生 `<select>`/`<input>` 的浏览器 chrome、四边不一致的 bevel 边框等问题）：

| 用途 | 组件 |
|------|------|
| 文本/数字输入、文本域 | `NInput` / `NInputNumber` |
| 下拉选择 | `NSelect` |
| 气泡浮层 | `NPopover` |
| 取色 | `NColorPicker` |
| 弹窗 / 对话框 | `NModal` / `NDialog`（按需引入） |
| 数据表格 | `NDataTable` |
| 可拖拽分栏 | `NSplit`（通过 `WorkspaceSplit` 统一使用） |
| 滚动容器 | `NScrollbar`（按需；简单容器可用全局细滚动条样式） |
| 虚拟长列表 | `NVirtualList`（列表很长时） |

- 主题：在 `App.vue` 用 `NConfigProvider` 包裹，主题由 `useNaiveTheme()` 依据 `--accent` + light/dark 生成（`themeOverrides` 与本项目 token 对齐：primaryColor=主题色、borderRadius=6px、输入框四边同色细边框、表格/弹层背景=surface 等）。
- 按钮统一用 `NButton`；标签仍用 `chip`；复杂控件继续用 Naive UI。
- `index.html` 有 `<meta name="naive-ui-style" />` 锚点，保证与 UnoCSS 注入顺序确定。

## 10. 滚动与手势

- 外层框架固定：`html` / `body` / `#app` / `.app-shell` / `.app-workspace` 不允许滚动，统一 `overflow: hidden` + `overscroll-behavior: none`。
- 只有明确的内部容器可滚动，例如消息主线、列表内容、设置页内容等使用 `overflow-y-auto` / `overflow-auto`；这些容器使用 `overscroll-behavior-y: contain`，滚动到边界时不把弹性滚动传到外层页面。
- 触控板双指横向滑动通过 `usePreventNavigationSwipe()` 兜底拦截：当事件不是内部横向滚动容器可消费的滚动时，阻止默认行为，降低误触浏览器前进/后退。

## 11. 落地位置

- 颜色与全局样式：`frontend/src/styles/theme.css`
- 组件级 shortcuts/rules：`frontend/uno.config.ts`（保留 `chip`、`card`、`input`、`bg-accent-soft`、`shadow-pop` 等；按钮不再使用 `btn-*` / `tbtn`）
- 主题切换逻辑：`frontend/src/composables/useTheme.ts`（light/dark/system + `--accent`，导出 `useIsDark`）
- Naive UI 主题：`frontend/src/composables/useNaiveTheme.ts` + `App.vue` 的 `NConfigProvider`
- 手势保护：`frontend/src/composables/usePreventNavigationSwipe.ts`

## 12. 设置页细节

- 主题（浅色 / 深色 / 跟随系统）使用 `NTabs type="segment"`，必须能清晰显示当前选中状态。
- 主题色预设色板与自定义取色器分开布局；取色器宽度不能压缩到只显示残缺 hex 文本。
- `API base`、`API resource` 等左侧 label 使用固定宽度与 `white-space: nowrap`，不要换行；右侧输入框保持紧凑宽度。
