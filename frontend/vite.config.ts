import { copyFileSync, existsSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, type Plugin } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import UnoCSS from 'unocss/vite'

// GitHub Pages 对 history 路由（如硬刷新 /callback）会回退到 404.html。
// 构建后把 index.html 复制为 404.html，让 SPA 接管所有路径。
function spaFallback(): Plugin {
  return {
    name: 'spa-404-fallback',
    apply: 'build',
    closeBundle() {
      const dist = fileURLToPath(new URL('./dist', import.meta.url))
      const index = `${dist}/index.html`
      if (existsSync(index)) copyFileSync(index, `${dist}/404.html`)
    },
  }
}

// 本地开发把 /api 代理到后端，避免 CORS；生产用 VITE_API_BASE 指向 funmg.dp.tech/sci-loop-api。
// 注意：Logto 重定向 URI 注册的是 http://localhost:9527/callback，dev server 必须跑在 9527。
const API_TARGET = process.env.VITE_API_TARGET || 'http://localhost:8080'

export default defineConfig({
  // GitHub Pages 自定义域名（rag.hal9k.one）部署在根路径，base 用 '/'。
  base: process.env.VITE_BASE || '/',
  plugins: [vue(), UnoCSS(), spaFallback()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 9527,
    strictPort: true,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
