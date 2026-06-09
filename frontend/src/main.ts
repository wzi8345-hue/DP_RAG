import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createLogto } from '@logto/vue'
import { enableKatex } from 'markstream-vue'

import 'virtual:uno.css'
import 'markstream-vue/index.css'
import 'katex/dist/katex.min.css'
import '@/styles/theme.css'

// 启用 KaTeX 数学公式渲染（科学内容需要）。
enableKatex()

import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import { logtoConfig } from './auth/logto'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(i18n)
app.use(createLogto, logtoConfig)

app.mount('#app')
