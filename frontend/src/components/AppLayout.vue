<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { useLogto } from '@logto/vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { NButton } from 'naive-ui'
import { postLogoutRedirectUri, redirectUri } from '@/auth/logto'
import { useAuthz } from '@/composables/useAuthz'
import UserMenuPopover from '@/components/UserMenuPopover.vue'

type UserInfo = {
  email?: string | null
  name?: string | null
  username?: string | null
  sub?: string | null
}

const { t } = useI18n()
const router = useRouter()
const { signOut, signIn, fetchUserInfo, isAuthenticated } = useLogto()
const authz = useAuthz()

const nav = computed(() => [
  { name: 'chat', to: '/chat', icon: 'i-lucide-messages-square', label: t('nav.chat') },
  { name: 'library', to: '/library', icon: 'i-lucide-library-big', label: t('nav.library') },
  { name: 'skills', to: '/skills', icon: 'i-lucide-puzzle', label: t('nav.skills') },
  ...(authz.isAdmin.value
    ? [{ name: 'admin', to: '/admin', icon: 'i-lucide-shield-check', label: t('nav.admin') }]
    : []),
])

function displayValue(value?: string | null) {
  const trimmed = value?.trim()
  if (!trimmed || trimmed === userInfo.value.sub) return ''
  return trimmed
}

const userInfo = shallowRef<UserInfo>({})
const userEmail = computed(() => displayValue(userInfo.value.email))
const userName = computed(() => displayValue(userInfo.value.name) || displayValue(userInfo.value.username))

function openSettings() {
  router.push({ path: '/settings', hash: '#general' })
}

function handleSignOut() {
  signOut(postLogoutRedirectUri)
}

// 仅在已登录时拉取用户信息；用 watch(immediate) 而非 onMounted，避免未登录时
// 发起 Logto 请求，也保证登录态从 false→true 时（无需重挂组件）补取一次。
watch(
  isAuthenticated,
  async (authed) => {
    if (!authed) {
      userInfo.value = {}
      return
    }
    try {
      userInfo.value = (await fetchUserInfo()) || {}
    } catch {
      userInfo.value = {}
    }
  },
  { immediate: true },
)
</script>

<template>
  <div class="app-shell flex h-full w-full overflow-hidden text-base">
    <aside class="primary-sidebar">
      <div class="flex items-center gap-2 px-3 py-3">
        <div class="grid h-8 w-8 place-items-center rounded-[7px] bg-accent text-xs font-bold text-white">
          DP
        </div>
        <div class="min-w-0">
          <div class="truncate text-sm font-semibold leading-tight">{{ t('app.name') }}</div>
          <div class="truncate text-xs text-faint">{{ t('app.tagline') }}</div>
        </div>
      </div>

      <nav class="flex flex-1 flex-col gap-0.5">
        <RouterLink
          v-for="n in nav"
          :key="n.name"
          :to="n.to"
          class="flex items-center gap-2.5 rounded-[6px] px-2.5 py-2 text-sm text-muted transition-colors hover:bg-hover hover:text-base"
          active-class="!bg-active !text-base"
        >
          <span :class="n.icon" class="text-[1.05rem]" />
          {{ n.label }}
        </RouterLink>
      </nav>

      <div class="p-1.5">
        <UserMenuPopover
          v-if="isAuthenticated"
          :email="userEmail"
          :name="userName"
          @settings="openSettings"
          @sign-out="handleSignOut"
        />
        <NButton v-else type="primary" block @click="signIn(redirectUri)">
          <template #icon>
            <span class="i-lucide-log-in" />
          </template>
          {{ t('auth.signIn') }}
        </NButton>
      </div>
    </aside>

    <main class="app-workspace">
      <RouterView v-if="isAuthenticated" />

      <div v-else class="h-full grid place-items-center bg-app px-4">
        <div class="card w-full max-w-sm p-7 text-center">
          <div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-[8px] bg-accent text-lg font-bold text-white">
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
    </main>
  </div>
</template>
