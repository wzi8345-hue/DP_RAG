<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useLogto } from '@logto/vue'
import { useI18n } from 'vue-i18n'
import { NButton, NConfigProvider } from 'naive-ui'
import { useTheme } from '@/composables/useTheme'
import { useNaiveTheme } from '@/composables/useNaiveTheme'
import { usePreventNavigationSwipe } from '@/composables/usePreventNavigationSwipe'
import { redirectUri } from '@/auth/logto'
import AppLayout from '@/components/AppLayout.vue'

useTheme()
usePreventNavigationSwipe()
const { theme: naiveTheme, themeOverrides: naiveOverrides } = useNaiveTheme()
const route = useRoute()
const { isAuthenticated, isLoading, signIn } = useLogto()
const { t } = useI18n()

const isPublic = computed(() => route.meta.public === true)

// 关键修复：@logto/vue 的 isLoading 不只是"首次鉴权检查"，它在每次
// getAccessToken()/fetchUserInfo() 时都会切到 true。若直接用 isLoading 作为
// 渲染门控，AppLayout 挂载后触发的 token/userinfo 请求会把 isLoading 翻成 true →
// AppLayout 被卸载 → 请求 resolve 后 isLoading=false → 重新挂载 → 再次发请求…
// 形成"挂载/卸载"死循环，持续向后端发请求。
// 因此只用 isLoading 锁存"首次鉴权是否完成"一次，之后只按 isAuthenticated 门控。
const initialAuthChecked = ref(false)
watch(
  isLoading,
  (loading) => {
    if (!loading) initialAuthChecked.value = true
  },
  { immediate: true },
)
</script>

<template>
  <NConfigProvider :theme="naiveTheme" :theme-overrides="naiveOverrides" class="h-full">
    <RouterView v-if="isPublic" />

    <div v-else-if="!initialAuthChecked" class="h-full grid place-items-center bg-app text-muted">
      <div class="flex items-center gap-2 text-sm">
        <span class="i-lucide-loader-circle animate-spin" /> {{ t('common.loading') }}
      </div>
    </div>

    <div v-else-if="!isAuthenticated" class="h-full grid place-items-center bg-app px-4">
      <div class="card w-full max-w-sm p-7 text-center">
        <div class="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-[8px] bg-accent text-base font-bold text-white">
          DP
        </div>
        <h1 class="text-lg font-semibold text-base">{{ t('app.name') }}</h1>
        <p class="mt-1 text-sm text-muted">{{ t('app.tagline') }}</p>
        <p class="mt-4 text-xs text-faint">{{ t('auth.required') }}</p>
        <NButton type="primary" block class="mt-5" @click="signIn(redirectUri)">
          <template #icon>
            <span class="i-lucide-log-in" />
          </template>
          {{ t('auth.signIn') }}
        </NButton>
      </div>
    </div>

    <AppLayout v-else />
  </NConfigProvider>
</template>
