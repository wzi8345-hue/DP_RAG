<script setup lang="ts">
import { computed } from 'vue'
import type { ResearchMeta } from '@/api/types'

const props = defineProps<{ research: ResearchMeta }>()
const insufficient = computed(() => props.research.status === 'insufficient')
</script>

<template>
  <div class="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
    <span
      class="chip"
      :class="insufficient ? 'text-amber-600' : 'text-accent'"
    >
      <span class="i-lucide-graduation-cap" />
      {{ insufficient ? '专家模式 · 证据不足' : '专家模式' }}
    </span>
    <span class="chip">{{ research.rounds }} 轮检索</span>
    <span class="chip">{{ research.evidence_docs }} 篇 · {{ research.evidence_chunks }} 证据</span>
    <span
      v-if="research.gaps && research.gaps.length > 0"
      class="chip text-amber-600"
      :title="research.gaps.join('\n')"
    >
      <span class="i-lucide-triangle-alert" /> {{ research.gaps.length }} 缺口
    </span>
  </div>
</template>
