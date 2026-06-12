<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useLogto } from '@logto/vue'
import { useI18n } from 'vue-i18n'
import { NConfigProvider } from 'naive-ui'
import { useTheme } from '@/composables/useTheme'
import { useNaiveTheme } from '@/composables/useNaiveTheme'
import { usePreventNavigationSwipe } from '@/composables/usePreventNavigationSwipe'
import AppLayout from '@/components/AppLayout.vue'

useTheme()
usePreventNavigationSwipe()
const { theme: naiveTheme, themeOverrides: naiveOverrides } = useNaiveTheme()
const route = useRoute()
const { isLoading } = useLogto()
const { t } = useI18n()

const isPublic = computed(() => route.meta.public === true)

// 关键修复：@logto/vue 的 isLoading 不只是"首次鉴权检查"，它在每次
// getAccessToken()/fetchUserInfo() 时都会切到 true。若直接用 isLoading 作为
// 渲染门控，AppLayout 挂载后触发的 token/userinfo 请求会把 isLoading 翻成 true →
// AppLayout 被卸载 → 请求 resolve 后 isLoading=false → 重新挂载 → 再次发请求…
// 形成"挂载/卸载"死循环，持续向后端发请求。
// 因此只用 isLoading 锁存"首次鉴权是否完成"一次，之后只渲染 AppLayout（不再随
// 登录态卸载/重挂）。是否登录由 AppLayout 内部按 isAuthenticated 决定展示登录入口
// 还是业务页面（未登录时不挂载任何会发后端请求的页面，避免 401 风暴 / 死循环）。
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

    <AppLayout v-else />
  </NConfigProvider>
</template>
