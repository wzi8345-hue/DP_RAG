import { useLogto } from '@logto/vue'
import { ApiClient } from '@/api/client'
import { API_RESOURCE } from '@/auth/logto'
import { useSettings } from './useSettings'

let client: ApiClient | null = null

/**
 * 返回单例 ApiClient，token 走 Logto getAccessToken(API_RESOURCE)，
 * base 实时读取设置。必须在组件 setup 内调用（依赖 useLogto）。
 */
export function useApi(): ApiClient {
  const { getAccessToken, isAuthenticated } = useLogto()
  const settings = useSettings()

  if (!client) {
    client = new ApiClient({
      base: () => settings.value.apiBase,
      getToken: async () => {
        if (!isAuthenticated.value) return undefined
        try {
          return await getAccessToken(API_RESOURCE)
        } catch {
          return undefined
        }
      },
    })
  }
  return client
}
