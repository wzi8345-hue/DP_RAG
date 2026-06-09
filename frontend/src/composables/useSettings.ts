import { useStorage } from '@vueuse/core'
import type { RemovableRef } from '@vueuse/core'
import type { RetrievalMode, RetrievalSourceKey } from '@/api/types'

export type ThemeMode = 'light' | 'dark' | 'system'

export interface AppSettings {
  /** 后端 API base；空串=同源（dev proxy）。默认取构建期 VITE_API_BASE。 */
  apiBase: string
  theme: ThemeMode
  accent: string
  // 检索默认
  mode: RetrievalMode | 'auto'
  topK: number
  useAgentic: boolean
  stream: boolean
  professional: boolean
  enableRetrieval: boolean
  sources: RetrievalSourceKey[]
}

const DEFAULTS: AppSettings = {
  apiBase: import.meta.env.VITE_API_BASE || '',
  theme: 'system',
  accent: '#2563eb',
  mode: 'auto',
  topK: 5,
  useAgentic: true,
  stream: true,
  professional: false,
  enableRetrieval: true,
  sources: ['literature'],
}

let singleton: RemovableRef<AppSettings> | null = null

export function useSettings(): RemovableRef<AppSettings> {
  if (!singleton) {
    singleton = useStorage<AppSettings>('dp-rag-settings', DEFAULTS, localStorage, {
      mergeDefaults: true,
    })
  }
  return singleton
}
