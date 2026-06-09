import { computed } from 'vue'
import { darkTheme, type GlobalTheme, type GlobalThemeOverrides } from 'naive-ui'
import { useSettings } from './useSettings'
import { useIsDark } from './useTheme'

// 把 #rrggbb 与白/黑按权重混合，派生 hover/pressed 等态（Naive 需要实色，不能用 var()）
function mix(hex: string, target: string, weight: number): string {
  const a = hexToRgb(hex)
  const b = hexToRgb(target)
  if (!a || !b) return hex
  const r = Math.round(a[0] + (b[0] - a[0]) * weight)
  const g = Math.round(a[1] + (b[1] - a[1]) * weight)
  const bl = Math.round(a[2] + (b[2] - a[2]) * weight)
  return `#${[r, g, bl].map((n) => n.toString(16).padStart(2, '0')).join('')}`
}

function hexToRgb(hex: string): [number, number, number] | null {
  let h = hex.replace('#', '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  if (h.length !== 6) return null
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

// 与 theme.css 保持一致的两套基础色板（Naive 主题用实色）
const LIGHT = {
  bg: '#ffffff',
  surface: '#ffffff',
  surface2: '#f7f8fa',
  popover: '#ffffff',
  hover: '#f1f3f5',
  active: '#eceff3',
  border: '#e4e7eb',
  text: '#18181b',
  textMuted: '#6b7280',
  textFaint: '#9ca3af',
}
const DARK = {
  bg: '#0a0a0b',
  surface: '#131316',
  surface2: '#1a1a1e',
  popover: '#202127',
  hover: '#25252b',
  active: '#303039',
  border: '#303038',
  text: '#f4f4f5',
  textMuted: '#a1a1aa',
  textFaint: '#6b7280',
}

/** 返回与本应用主题（accent + light/dark）一致的 Naive UI 主题配置。 */
export function useNaiveTheme() {
  const settings = useSettings()
  const isDark = useIsDark()

  const theme = computed<GlobalTheme | null>(() => (isDark.value ? darkTheme : null))

  const themeOverrides = computed<GlobalThemeOverrides>(() => {
    const p = isDark.value ? DARK : LIGHT
    const rawAccent = settings.value.accent
    const accent = isDark.value ? mix(rawAccent, '#ffffff', 0.38) : rawAccent
    const hover = mix(accent, isDark.value ? '#ffffff' : '#ffffff', 0.12)
    const pressed = mix(accent, isDark.value ? '#ffffff' : '#000000', isDark.value ? 0.18 : 0.12)
    return {
      common: {
        primaryColor: accent,
        primaryColorHover: hover,
        primaryColorPressed: pressed,
        primaryColorSuppl: accent,
        borderRadius: '6px',
        borderRadiusSmall: '6px',
        fontSize: '14px',
        fontSizeSmall: '13px',
        // 基础色，与我们的 token 对齐
        bodyColor: p.bg,
        cardColor: p.surface,
        modalColor: p.surface,
        popoverColor: p.popover,
        tableColor: p.surface,
        inputColor: p.surface,
        inputColorDisabled: p.surface2,
        borderColor: p.border,
        dividerColor: p.border,
        hoverColor: p.hover,
        textColorBase: p.text,
        textColor1: p.text,
        textColor2: p.textMuted,
        textColor3: p.textFaint,
        placeholderColor: p.textFaint,
        tableHeaderColor: p.surface,
        scrollbarColor: isDark.value ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.18)',
        scrollbarColorHover: p.textFaint,
      },
      Button: {
        borderRadiusSmall: '6px',
        borderRadiusMedium: '6px',
        heightTiny: '22px',
        heightSmall: '28px',
        heightMedium: '34px',
        fontSizeTiny: '12px',
        fontSizeSmall: '12px',
        border: 'none',
        borderHover: 'none',
        borderPressed: 'none',
        borderFocus: 'none',
        borderDisabled: 'none',
        borderPrimary: 'none',
        borderHoverPrimary: 'none',
        borderPressedPrimary: 'none',
        borderFocusPrimary: 'none',
        borderError: 'none',
        borderHoverError: 'none',
        borderPressedError: 'none',
        borderFocusError: 'none',
        color: 'transparent',
        colorHover: p.hover,
        colorPressed: p.active,
        colorFocus: p.hover,
        colorTertiary: 'transparent',
        colorTertiaryHover: p.hover,
        colorTertiaryPressed: p.active,
        colorQuaternary: 'transparent',
        colorQuaternaryHover: p.hover,
        colorQuaternaryPressed: p.active,
        colorPrimary: accent,
        colorHoverPrimary: hover,
        colorPressedPrimary: pressed,
        colorFocusPrimary: hover,
        colorError: '#dc2626',
        colorHoverError: '#ef4444',
        colorPressedError: '#b91c1c',
        colorFocusError: '#ef4444',
        textColor: p.textMuted,
        textColorHover: p.text,
        textColorPressed: p.text,
        textColorFocus: p.text,
        textColorTertiary: p.textMuted,
        textColorText: p.textMuted,
        textColorTextHover: accent,
        textColorTextPressed: accent,
        textColorTextFocus: accent,
        textColorPrimary: '#ffffff',
        textColorHoverPrimary: '#ffffff',
        textColorPressedPrimary: '#ffffff',
        textColorFocusPrimary: '#ffffff',
        textColorError: '#ffffff',
        textColorHoverError: '#ffffff',
        textColorPressedError: '#ffffff',
        textColorFocusError: '#ffffff',
        opacityDisabled: '0.48',
        rippleColor: accent,
        rippleColorPrimary: accent,
      },
      Input: {
        // 四边一致的细边框（消除原生输入框的 3D bevel/上下左右不一致）
        border: `1px solid ${p.border}`,
        borderHover: `1px solid ${accent}`,
        borderFocus: `1px solid ${accent}`,
        boxShadowFocus: 'none',
        color: p.surface,
      },
      Select: {
        peers: {
          InternalSelection: {
            border: `1px solid ${p.border}`,
            borderHover: `1px solid ${accent}`,
            borderActive: `1px solid ${accent}`,
            borderFocus: `1px solid ${accent}`,
            boxShadowActive: 'none',
            boxShadowFocus: 'none',
          },
        },
      },
      DataTable: {
        borderColor: p.border,
        thColor: p.surface,
        tdColor: p.surface,
        tdColorHover: p.hover,
      },
      Tabs: {
        colorSegment: p.surface2,
        tabColorSegment: p.active,
        tabTextColorSegment: p.textMuted,
        tabTextColorActiveSegment: p.text,
        tabTextColorHoverSegment: p.text,
        tabBorderRadius: '6px',
        tabFontWeightActive: '600',
        tabPaddingSmallSegment: '0 10px',
      },
      Split: {
        resizableTriggerColor: p.border,
        resizableTriggerColorHover: p.active,
      },
    }
  })

  return { theme, themeOverrides }
}
