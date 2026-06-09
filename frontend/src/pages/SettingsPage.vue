<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useLogto } from '@logto/vue'
import { NButton, NCheckbox, NColorPicker, NInput, NInputNumber, NSelect, NTabPane, NTabs } from 'naive-ui'
import { useSettings, type ThemeMode } from '@/composables/useSettings'
import { setLocale, type AppLocale } from '@/i18n'
import { postLogoutRedirectUri, API_RESOURCE } from '@/auth/logto'
import type { RetrievalMode } from '@/api/types'

const { t, locale } = useI18n()
const settings = useSettings()
const { signOut, fetchUserInfo } = useLogto()

const themes: { value: ThemeMode; label: string; icon: string }[] = [
  { value: 'light', label: 'settings.themeLight', icon: 'i-lucide-sun' },
  { value: 'dark', label: 'settings.themeDark', icon: 'i-lucide-moon' },
  { value: 'system', label: 'settings.themeSystem', icon: 'i-lucide-monitor' },
]
const accents = ['#2563eb', '#16a34a', '#9333ea', '#e11d48', '#ea580c', '#0891b2', '#0a0a0a']
const modes: (RetrievalMode | 'auto')[] = ['auto', 'hybrid', 'vector', 'metadata']
const modeOptions = computed(() => modes.map((m) => ({ label: m, value: m })))

const userLabel = ref('')
function changeLocale(l: AppLocale) {
  setLocale(l)
  locale.value = l
}

function displayUserLabel(info: { email?: string | null; name?: string | null; username?: string | null; sub?: string | null } | undefined) {
  if (!info) return ''
  const candidates = [info.email, info.name, info.username]
  return candidates.map((value) => value?.trim()).find((value) => value && value !== info.sub) || ''
}

onMounted(async () => {
  try {
    const info = await fetchUserInfo()
    userLabel.value = displayUserLabel(info)
  } catch {
    /* ignore */
  }
})
</script>

<template>
  <div class="h-full overflow-y-auto">
    <div class="mx-auto max-w-2xl px-6 py-6">
      <h1 class="mb-5 text-lg font-semibold">{{ t('settings.title') }}</h1>

      <div class="card">
      <!-- 外观 -->
      <section class="p-4">
        <h2 class="mb-3 text-sm font-semibold text-base">{{ t('settings.appearance') }}</h2>
        <div id="theme" class="flex items-center justify-between py-2">
          <span class="text-sm text-muted">{{ t('settings.theme') }}</span>
          <NTabs
            v-model:value="settings.theme"
            type="segment"
            size="small"
            class="w-56"
            pane-class="hidden"
            :tabs-padding="0"
          >
            <NTabPane
              v-for="th in themes"
              :key="th.value"
              :name="th.value"
            >
              <template #tab>
                <span class="inline-flex items-center gap-1">
                <span :class="th.icon" />
                  {{ t(th.label) }}
                </span>
              </template>
            </NTabPane>
          </NTabs>
        </div>
        <div id="language" class="flex items-center justify-between py-2">
          <span class="text-sm text-muted">{{ t('settings.accent') }}</span>
          <div class="flex flex-wrap items-center justify-end gap-1.5">
            <NButton
              v-for="a in accents"
              :key="a"
              class="!h-5 !w-5 !min-w-5 rounded-full !p-0 transition"
              circle
              size="tiny"
              :bordered="false"
              :color="a"
              :style="settings.accent === a ? { background: a, outline: '2px solid var(--accent)', outlineOffset: '2px' } : { background: a }"
              @click="settings.accent = a"
            />
            <NColorPicker
              v-model:value="settings.accent"
              class="w-28"
              size="small"
              :show-alpha="false"
              :show-preview="true"
              :modes="['hex']"
              :swatches="accents"
            />
          </div>
        </div>
        <div class="flex items-center justify-between py-2">
          <span class="text-sm text-muted">{{ t('settings.language') }}</span>
          <div class="flex items-center gap-1 rounded-[6px] bg-surface p-0.5">
            <NButton
              :tertiary="locale === 'zh-CN'"
              :quaternary="locale !== 'zh-CN'"
              size="tiny"
              @click="changeLocale('zh-CN')"
            >中文</NButton>
            <NButton
              :tertiary="locale === 'en'"
              :quaternary="locale !== 'en'"
              size="tiny"
              @click="changeLocale('en')"
            >English</NButton>
          </div>
        </div>
      </section>

      <!-- 检索默认 -->
      <section id="general" class="border-t border-softer p-4">
        <h2 class="mb-3 text-sm font-semibold text-base">{{ t('settings.retrieval') }}</h2>
        <div class="flex items-center justify-between py-2">
          <span class="text-sm text-muted">{{ t('settings.defaultMode') }}</span>
          <NSelect v-model:value="settings.mode" :options="modeOptions" size="small" class="w-40" />
        </div>
        <div class="flex items-center justify-between py-2">
          <span class="text-sm text-muted">{{ t('settings.topK') }}</span>
          <NInputNumber v-model:value="settings.topK" :min="1" :max="20" size="small" class="w-28" />
        </div>
        <div class="py-2">
          <span class="text-sm text-muted">{{ t('settings.sources') }}</span>
          <div class="mt-2 flex flex-col gap-2">
            <NCheckbox
              :checked="settings.sources.includes('literature')"
              @update:checked="settings.sources = settings.sources.includes('literature') ? settings.sources.filter(s => s !== 'literature') : [...settings.sources, 'literature']"
            >
              {{ t('settings.sourceLiterature') }}
            </NCheckbox>
            <div class="flex items-center gap-2" :title="t('settings.sourceSqlSoon')">
              <NCheckbox disabled>{{ t('settings.sourceSql') }}</NCheckbox>
              <span class="chip">{{ t('settings.sourceSqlSoon') }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- 账号 -->
      <section class="border-t border-softer p-4">
        <h2 class="mb-3 text-sm font-semibold text-base">{{ t('settings.account') }}</h2>
        <div class="flex items-center justify-between py-1">
          <span class="text-sm text-muted">{{ userLabel || '—' }}</span>
          <NButton tertiary size="small" @click="signOut(postLogoutRedirectUri)">
            <template #icon>
              <span class="i-lucide-log-out" />
            </template>
            {{ t('auth.signOut') }}
          </NButton>
        </div>
      </section>

      <!-- 高级 / 关于 -->
      <section class="border-t border-softer p-4">
        <h2 class="mb-3 text-sm font-semibold text-base">{{ t('settings.about') }}</h2>
          <div class="flex flex-col gap-1.5 text-xs text-faint">
          <div class="flex items-center justify-between gap-3">
            <span class="w-24 shrink-0 whitespace-nowrap">API base</span>
            <NInput v-model:value="settings.apiBase" size="small" class="w-56" placeholder="https://funmg.dp.tech/sci-loop-api" />
          </div>
          <div class="flex items-center justify-between gap-3">
            <span class="w-24 shrink-0 whitespace-nowrap">API resource</span>
            <span class="truncate">{{ API_RESOURCE }}</span>
          </div>
          <div class="flex items-center justify-between gap-3">
            <span class="w-24 shrink-0 whitespace-nowrap">{{ t('app.name') }}</span>
            <span>v2.0.0</span>
          </div>
        </div>
      </section>
      </div>
    </div>
  </div>
</template>
