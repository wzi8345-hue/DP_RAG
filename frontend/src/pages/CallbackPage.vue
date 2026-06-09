<script setup lang="ts">
import { useHandleSignInCallback } from '@logto/vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'

const router = useRouter()
const { t } = useI18n()

const { isLoading, error } = useHandleSignInCallback(() => {
  router.replace('/chat')
})
</script>

<template>
  <div class="h-full grid place-items-center bg-app text-muted">
    <div v-if="error" class="card max-w-sm p-6 text-center">
      <span class="i-lucide-circle-alert text-red-500" />
      <p class="mt-2 text-sm text-base">{{ t('auth.callbackError') }}</p>
      <p class="mt-1 text-xs text-faint">{{ error.message }}</p>
      <NButton tertiary class="mt-4" @click="router.replace('/')">{{ t('common.retry') }}</NButton>
    </div>
    <div v-else-if="isLoading" class="flex items-center gap-2 text-sm">
      <span class="i-lucide-loader-circle animate-spin" /> {{ t('auth.signingIn') }}
    </div>
  </div>
</template>
