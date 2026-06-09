<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NPopover } from 'naive-ui'
import { setLocale, type AppLocale } from '@/i18n'
import { useSettings, type ThemeMode } from '@/composables/useSettings'

const props = defineProps<{
  email: string
  name: string
}>()

const emit = defineEmits<{
  settings: []
  signOut: []
}>()

const { t, locale } = useI18n()
const settings = useSettings()

const primaryLabel = computed(() => props.email || props.name || t('auth.account'))
const secondaryLabel = computed(() => (props.email && props.name ? props.name : t('app.tagline')))
const initial = computed(() => primaryLabel.value.trim().slice(0, 1).toUpperCase() || 'U')

const languageOptions: { value: AppLocale; label: string }[] = [
  { value: 'zh-CN', label: '中文' },
  { value: 'en', label: 'English' },
]
const themeOptions: { value: ThemeMode; label: string; icon: string }[] = [
  { value: 'light', label: 'settings.themeLight', icon: 'i-lucide-sun' },
  { value: 'dark', label: 'settings.themeDark', icon: 'i-lucide-moon' },
  { value: 'system', label: 'settings.themeSystem', icon: 'i-lucide-monitor' },
]
const currentLanguageLabel = computed(() => (
  languageOptions.find((option) => option.value === locale.value)?.label || languageOptions[0].label
))
const currentThemeLabel = computed(() => (
  themeOptions.find((option) => option.value === settings.value.theme)?.label || themeOptions[0].label
))

function changeLocale(nextLocale: AppLocale) {
  setLocale(nextLocale)
  locale.value = nextLocale
}
</script>

<template>
  <NPopover
    trigger="click"
    placement="right-end"
    :show-arrow="false"
    :overlap="false"
    style="padding: 0; border-radius: 12px;"
  >
    <template #trigger>
      <NButton quaternary block class="user-menu-trigger">
        <span class="user-menu-avatar">{{ initial }}</span>
        <span class="min-w-0 flex-1 text-left">
          <span class="block truncate text-sm font-medium text-base" :title="primaryLabel">{{ primaryLabel }}</span>
          <span class="block truncate text-xs text-faint" :title="secondaryLabel">{{ secondaryLabel }}</span>
        </span>
        <span class="i-lucide-chevrons-up-down shrink-0 text-faint text-[0.95rem]" />
      </NButton>
    </template>

    <div class="user-menu-panel">
      <div class="user-menu-profile">
        <span class="user-menu-avatar size-lg">{{ initial }}</span>
        <span class="min-w-0">
          <span class="block truncate text-sm font-medium text-base" :title="primaryLabel">{{ primaryLabel }}</span>
          <span class="block truncate text-xs text-faint" :title="secondaryLabel">{{ secondaryLabel }}</span>
        </span>
      </div>

      <div class="menu-section-divider" />

      <div class="user-menu-group">
        <NPopover
          trigger="click"
          placement="right-start"
          :show-arrow="false"
          :overlap="false"
          style="padding: 0; border-radius: 10px;"
        >
          <template #trigger>
            <NButton quaternary block class="user-menu-item has-value">
              <span class="i-lucide-languages shrink-0 text-[1rem]" />
              <span class="user-menu-label">{{ t('settings.languagePreference') }}</span>
              <span class="user-menu-value">{{ currentLanguageLabel }}</span>
              <span class="i-lucide-chevron-right shrink-0 text-faint text-[0.9rem]" />
            </NButton>
          </template>

          <div class="user-submenu-panel">
            <NButton
              v-for="option in languageOptions"
              :key="option.value"
              quaternary
              block
              class="user-submenu-item"
              @click="changeLocale(option.value)"
            >
              <span class="user-submenu-check">
                <span v-if="locale === option.value" class="i-lucide-check" />
              </span>
              <span class="user-submenu-label">{{ option.label }}</span>
            </NButton>
          </div>
        </NPopover>

        <NPopover
          trigger="click"
          placement="right-start"
          :show-arrow="false"
          :overlap="false"
          style="padding: 0; border-radius: 10px;"
        >
          <template #trigger>
            <NButton quaternary block class="user-menu-item has-value">
              <span class="i-lucide-palette shrink-0 text-[1rem]" />
              <span class="user-menu-label">{{ t('settings.themePreference') }}</span>
              <span class="user-menu-value">{{ t(currentThemeLabel) }}</span>
              <span class="i-lucide-chevron-right shrink-0 text-faint text-[0.9rem]" />
            </NButton>
          </template>

          <div class="user-submenu-panel">
            <NButton
              v-for="option in themeOptions"
              :key="option.value"
              quaternary
              block
              class="user-submenu-item with-icon"
              @click="settings.theme = option.value"
            >
              <span class="user-submenu-check">
                <span v-if="settings.theme === option.value" class="i-lucide-check" />
              </span>
              <span :class="option.icon" class="shrink-0 text-[0.95rem]" />
              <span class="user-submenu-label">{{ t(option.label) }}</span>
            </NButton>
          </div>
        </NPopover>

        <NButton quaternary block class="user-menu-item simple" @click="emit('settings')">
          <span class="i-lucide-settings-2 shrink-0 text-[1rem]" />
          <span class="user-menu-label">{{ t('settings.generalSettings') }}</span>
        </NButton>
      </div>

      <div class="menu-section-divider" />

      <NButton quaternary block class="user-menu-item simple danger" @click="emit('signOut')">
        <span class="i-lucide-log-out shrink-0 text-[1rem]" />
        <span class="user-menu-label">{{ t('auth.signOut') }}</span>
      </NButton>
    </div>
  </NPopover>
</template>
