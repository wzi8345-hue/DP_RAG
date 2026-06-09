<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'

defineProps<{ index: number; total: number }>()
const emit = defineEmits<{ (e: 'prev'): void; (e: 'next'): void }>()
const { t } = useI18n()
</script>

<template>
  <span class="inline-flex items-center gap-0.5 text-xs">
    <NButton
      text
      size="tiny"
      :disabled="index <= 0"
      :title="t('chat.branchPrev')"
      @click="emit('prev')"
    >
      <template #icon>
        <span class="i-lucide-chevron-left" />
      </template>
    </NButton>
    <span class="tabular-nums">{{ index + 1 }}/{{ total }}</span>
    <NButton
      text
      size="tiny"
      :disabled="index >= total - 1"
      :title="t('chat.branchNext')"
      @click="emit('next')"
    >
      <template #icon>
        <span class="i-lucide-chevron-right" />
      </template>
    </NButton>
  </span>
</template>
