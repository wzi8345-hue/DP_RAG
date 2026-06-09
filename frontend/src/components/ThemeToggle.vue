<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'
import { useSettings, type ThemeMode } from '@/composables/useSettings'

const settings = useSettings()
const { t } = useI18n()

const order: ThemeMode[] = ['system', 'light', 'dark']
const icon = computed(() => ({
  system: 'i-lucide-monitor',
  light: 'i-lucide-sun',
  dark: 'i-lucide-moon',
}[settings.value.theme]))

const label = computed(() => ({
  system: t('settings.themeSystem'),
  light: t('settings.themeLight'),
  dark: t('settings.themeDark'),
}[settings.value.theme]))

function cycle() {
  const i = order.indexOf(settings.value.theme)
  settings.value.theme = order[(i + 1) % order.length]
}
</script>

<template>
  <NButton quaternary circle size="small" :title="`${t('settings.theme')}: ${label}`" @click="cycle">
    <template #icon>
      <span :class="icon" />
    </template>
  </NButton>
</template>
