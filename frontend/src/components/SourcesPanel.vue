<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import type { DocSummaryResponse, Hit } from '@/api/types'

export interface SourceItem {
  num: number
  hit: Hit
}

const props = defineProps<{
  items: SourceItem[]
  tab: 'summary' | 'chunks'
  highlightNum?: number | null
}>()
const emit = defineEmits<{
  (e: 'close'): void
  (e: 'update:tab', v: 'summary' | 'chunks'): void
}>()

const { t } = useI18n()
const api = useApi()
const summaries = ref<Record<string, DocSummaryResponse | 'loading' | 'error'>>({})

const docs = computed(() => {
  const seen = new Map<string, { num: number; hit: Hit }>()
  for (const it of props.items) {
    const key = it.hit.doc_id || it.hit.doc_name || String(it.num)
    if (!seen.has(key)) seen.set(key, it)
  }
  return [...seen.values()]
})

async function loadSummary(docId: string) {
  if (!docId || summaries.value[docId]) return
  summaries.value[docId] = 'loading'
  try {
    summaries.value[docId] = await api.docSummary(docId)
  } catch {
    summaries.value[docId] = 'error'
  }
}

watch(
  () => [props.tab, docs.value] as const,
  () => {
    if (props.tab === 'summary') {
      for (const d of docs.value) {
        const id = d.hit.doc_id || d.hit.doc_name
        if (id) loadSummary(id)
      }
    }
  },
  { immediate: true },
)
</script>

<template>
  <aside class="workspace-card ml-1 flex w-80 shrink-0 flex-col">
    <header class="flex items-center justify-between border-b border-softer px-4 py-3">
      <div class="flex items-center gap-1 text-sm">
        <NButton
          :tertiary="tab === 'summary'"
          :quaternary="tab !== 'summary'"
          size="tiny"
          @click="emit('update:tab', 'summary')"
        >
          {{ t('sources.summary') }}
        </NButton>
        <NButton
          :tertiary="tab === 'chunks'"
          :quaternary="tab !== 'chunks'"
          size="tiny"
          @click="emit('update:tab', 'chunks')"
        >
          {{ t('sources.chunks') }}
        </NButton>
      </div>
      <NButton quaternary circle size="small" :title="t('common.close')" @click="emit('close')">
        <template #icon>
          <span class="i-lucide-x" />
        </template>
      </NButton>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto p-3">
      <!-- 文献简介 -->
      <div v-if="tab === 'summary'" class="flex flex-col gap-2.5">
        <p v-if="docs.length === 0" class="py-8 text-center text-xs text-faint">{{ t('common.empty') }}</p>
        <div
          v-for="d in docs"
          :key="d.hit.doc_id || d.hit.doc_name || d.num"
          class="rounded-[8px] p-3 transition-colors"
          :class="highlightNum === d.num ? 'bg-accent-soft' : 'hover:bg-hover'"
        >
          <div class="flex items-start gap-2">
            <span class="cite-marker shrink-0">{{ d.num }}</span>
            <div class="min-w-0">
              <div class="truncate text-xs font-medium text-base" :title="d.hit.doc_name">
                {{ d.hit.doc_name || d.hit.doc_id }}
              </div>
              <template v-if="(d.hit.doc_id || d.hit.doc_name) && summaries[(d.hit.doc_id || d.hit.doc_name)!]">
                <p
                  v-if="summaries[(d.hit.doc_id || d.hit.doc_name)!] === 'loading'"
                  class="mt-1 text-xs text-faint"
                >
                  {{ t('common.loading') }}
                </p>
                <template v-else-if="typeof summaries[(d.hit.doc_id || d.hit.doc_name)!] === 'object'">
                  <div
                    class="mt-1 text-[11px] text-muted"
                  >
                    <span v-if="(summaries[(d.hit.doc_id || d.hit.doc_name)!] as DocSummaryResponse).year">
                      {{ t('sources.year', { n: (summaries[(d.hit.doc_id || d.hit.doc_name)!] as DocSummaryResponse).year }) }}
                    </span>
                  </div>
                  <p class="mt-1 line-clamp-6 text-xs leading-relaxed text-muted">
                    {{ (summaries[(d.hit.doc_id || d.hit.doc_name)!] as DocSummaryResponse).summary || t('sources.noSummary') }}
                  </p>
                </template>
              </template>
            </div>
          </div>
        </div>
      </div>

      <!-- 命中片段 -->
      <div v-else class="flex flex-col gap-2.5">
        <p v-if="items.length === 0" class="py-8 text-center text-xs text-faint">{{ t('common.empty') }}</p>
        <div
          v-for="it in items"
          :key="it.num"
          class="rounded-[8px] p-3 transition-colors"
          :class="highlightNum === it.num ? 'bg-accent-soft' : 'hover:bg-hover'"
        >
          <div class="mb-1 flex items-center gap-2">
            <span class="cite-marker shrink-0">{{ it.num }}</span>
            <span class="truncate text-xs font-medium text-base" :title="it.hit.doc_name">
              {{ it.hit.doc_name || it.hit.doc_id }}
            </span>
          </div>
          <div class="mb-1 flex flex-wrap gap-1 text-[10px] text-faint">
            <span v-if="it.hit.section" class="chip">{{ it.hit.section }}</span>
            <span v-if="it.hit.page_start != null" class="chip">{{ t('sources.page', { n: it.hit.page_start }) }}</span>
            <span v-if="it.hit.rerank_score != null" class="chip">{{ t('sources.rerank') }} {{ it.hit.rerank_score.toFixed(2) }}</span>
          </div>
          <p class="line-clamp-[12] whitespace-pre-wrap text-xs leading-relaxed text-muted">{{ it.hit.content }}</p>
        </div>
      </div>
    </div>
  </aside>
</template>
