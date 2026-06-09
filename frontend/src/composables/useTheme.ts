import { computed, watch, type ComputedRef } from 'vue'
import { usePreferredDark } from '@vueuse/core'
import { useSettings, type ThemeMode } from './useSettings'

/** 解析后的暗色判定：dark 或 (system 且系统暗色)。供 Naive UI 主题切换共用。 */
export function useIsDark(): ComputedRef<boolean> {
  const settings = useSettings()
  const prefersDark = usePreferredDark()
  return computed(
    () =>
      settings.value.theme === 'dark' ||
      (settings.value.theme === 'system' && prefersDark.value),
  )
}

/**
 * 应用主题到 <html>：light/dark/system + 自定义主题色（--accent）。
 * 在 App 根组件调用一次即可全局生效。
 */
export function useTheme() {
  const settings = useSettings()
  const isDark = useIsDark()

  function apply() {
    document.documentElement.classList.toggle('dark', isDark.value)
    document.documentElement.style.setProperty('--accent', settings.value.accent)
  }

  watch(
    [isDark, () => settings.value.accent],
    apply,
    { immediate: true },
  )

  function setTheme(mode: ThemeMode) {
    settings.value.theme = mode
  }
  function setAccent(color: string) {
    settings.value.accent = color
  }

  return { isDark, setTheme, setAccent }
}
