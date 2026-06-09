/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API base，如 https://funmg.dp.tech/sci-loop-api；空串=同源（走 dev proxy） */
  readonly VITE_API_BASE: string
  readonly VITE_LOGTO_ENDPOINT: string
  readonly VITE_LOGTO_APP_ID: string
  readonly VITE_LOGTO_RESOURCE: string
  readonly VITE_LOGTO_SCOPES: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}
