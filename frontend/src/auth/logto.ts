import type { LogtoConfig } from '@logto/vue'

/** 后端 API Resource（access_token 的 audience）。 */
export const API_RESOURCE = import.meta.env.VITE_LOGTO_RESOURCE || 'https://funmg.dp.tech/sci-loop-api'

const SCOPES = (import.meta.env.VITE_LOGTO_SCOPES || 'all:data,email,profile')
  .split(/[ ,]+/)
  .filter(Boolean)

export const logtoConfig: LogtoConfig = {
  endpoint: import.meta.env.VITE_LOGTO_ENDPOINT || 'https://auth.dplink.cc/',
  appId: import.meta.env.VITE_LOGTO_APP_ID || 'skjc9b4p12ykvz40vshjc',
  resources: [API_RESOURCE],
  scopes: SCOPES,
}

export const redirectUri = `${window.location.origin}/callback`
export const postLogoutRedirectUri = window.location.origin
