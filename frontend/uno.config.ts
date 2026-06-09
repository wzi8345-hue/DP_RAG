import {
  defineConfig,
  presetIcons,
  presetWind3,
  transformerDirectives,
  transformerVariantGroup,
} from 'unocss'

// 风格：类 Vercel 控制台。颜色全部走 CSS 变量 token（见 src/styles/theme.css），
// 这样 light/dark/system 与自定义主题色只需切换变量，组件无需改类名。
export default defineConfig({
  presets: [
    presetWind3(),
    presetIcons({
      scale: 1.1,
      warn: true,
      extraProperties: {
        'display': 'inline-block',
        'vertical-align': 'middle',
      },
      // 显式提供集合 loader，避免不同环境（pnpm 严格 node_modules）下自动发现失败。
      collections: {
        lucide: () =>
          import('@iconify-json/lucide/icons.json').then((m) => m.default as never),
      },
    }),
  ],
  transformers: [transformerDirectives(), transformerVariantGroup()],
  // 语义化颜色 → CSS 变量，便于主题切换
  rules: [
    [/^bg-app$/, () => ({ background: 'var(--bg)' })],
    [/^bg-surface$/, () => ({ background: 'var(--surface)' })],
    [/^bg-surface-2$/, () => ({ background: 'var(--surface-2)' })],
    [/^bg-hover$/, () => ({ background: 'var(--hover)' })],
    [/^bg-active$/, () => ({ background: 'var(--active)' })],
    [/^bg-active-hover$/, () => ({ background: 'var(--active-hover)' })],
    [/^text-base$/, () => ({ color: 'var(--text)' })],
    [/^text-muted$/, () => ({ color: 'var(--text-muted)' })],
    [/^text-faint$/, () => ({ color: 'var(--text-faint)' })],
    [/^text-accent$/, () => ({ color: 'var(--accent-visible)' })],
    [/^bg-accent$/, () => ({ background: 'var(--accent-visible)' })],
    [/^bg-accent-soft$/, () => ({ background: 'var(--accent-soft)' })],
    [/^bg-accent-soft-hover$/, () => ({ background: 'var(--accent-soft-hover)' })],
    [/^border-soft$/, () => ({ 'border-color': 'var(--border)' })],
    [/^border-softer$/, () => ({ 'border-color': 'var(--border-subtle)' })],
    [/^border-accent$/, () => ({ 'border-color': 'var(--accent-visible)' })],
    [/^ring-accent$/, () => ({ 'box-shadow': '0 0 0 1px var(--accent)' })],
  ],
  shortcuts: {
    // 容器：仅在确需独立成块时用（登录卡/浮层）；优先用分割线组织布局
    'card': 'bg-surface border border-soft rounded-[8px]',
    // 按钮：统一无 border、无 shadow；形态靠填充与 hover
    'btn': 'inline-flex items-center justify-center gap-1.5 rounded-[6px] px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed select-none',
    'btn-primary': 'btn bg-accent text-white hover:opacity-90',          // 主功能：纯色主题色填充
    'btn-secondary': 'btn bg-accent-soft text-accent hover:bg-accent-soft-hover', // 次要：主题色透明填充
    'btn-ghost': 'btn text-muted hover:bg-hover hover:text-base',        // 三级：无填充
    'btn-danger': 'btn text-red-600 hover:bg-red-500/10',               // 危险：红字红透明底
    'icon-btn': 'inline-flex items-center justify-center h-8 w-8 rounded-[6px] text-muted hover:bg-hover hover:text-base transition-colors',
    // 工具栏/控制按钮：无填充、无边框、无阴影；hover 才显中性底（Vercel 工具栏风）
    'tbtn': 'inline-flex items-center gap-1 rounded-[6px] px-2 py-1 text-xs text-muted transition-colors hover:bg-hover hover:text-base disabled:opacity-40 disabled:cursor-not-allowed select-none',
    'tbtn-on': 'bg-accent-soft text-accent hover:bg-accent-soft-hover hover:text-accent', // 工具按钮激活态
    'input': 'w-full rounded-[6px] border border-soft bg-surface px-3 py-1.5 text-sm text-base outline-none focus:border-accent placeholder:text-faint',
    'field-label': 'text-xs font-medium text-muted',
    'divider': 'border-t border-softer',
    // chip：纯标签用（非交互），无边框、次级底色
    'chip': 'inline-flex items-center gap-1 rounded-[6px] bg-surface-2 px-2 py-0.5 text-xs text-muted',
  },
})
