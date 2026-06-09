<script setup lang="ts">
import { computed, onMounted, shallowRef } from 'vue'
import { useLogto } from '@logto/vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { postLogoutRedirectUri } from '@/auth/logto'
import UserMenuPopover from '@/components/UserMenuPopover.vue'

type UserInfo = {
  email?: string | null
  name?: string | null
  username?: string | null
  sub?: string | null
}

const { t } = useI18n()
const router = useRouter()
const { signOut, fetchUserInfo } = useLogto()

const nav = computed(() => [
  { name: 'chat', to: '/chat', icon: 'i-lucide-messages-square', label: t('nav.chat') },
  { name: 'library', to: '/library', icon: 'i-lucide-library-big', label: t('nav.library') },
  { name: 'skills', to: '/skills', icon: 'i-lucide-puzzle', label: t('nav.skills') },
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

onMounted(async () => {
  try {
    userInfo.value = (await fetchUserInfo()) || {}
  } catch {
    userInfo.value = {}
  }
})
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
          :email="userEmail"
          :name="userName"
          @settings="openSettings"
          @sign-out="handleSignOut"
        />
      </div>
    </aside>

    <main class="app-workspace">
      <RouterView />
    </main>
  </div>
</template>
