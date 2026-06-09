<script setup lang="ts">
import { computed, ref, shallowRef, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'
import { PDFViewer, type PluginRegistry } from '@embedpdf/vue-pdf-viewer'
import { useApi } from '@/composables/useApi'
import type { DocSummaryResponse, Hit } from '@/api/types'

export interface SourceItem {
  num: number
  hit: Hit
}

const props = defineProps<{
  items: SourceItem[]
  tab: 'summary' | 'chunks' | 'pdf'
  highlightNum?: number | null
}>()
const emit = defineEmits<{
  (e: 'close'): void
  (e: 'update:tab', v: 'summary' | 'chunks' | 'pdf'): void
}>()

const { t } = useI18n()
const api = useApi()
const summaries = ref<Record<string, DocSummaryResponse | 'loading' | 'error'>>({})
const pdfUrl = ref('')
const pdfError = ref('')
const pdfLoading = ref(false)
const pdfDocKey = ref('')
const registry = shallowRef<PluginRegistry | null>(null)

const docs = computed(() => {
  const seen = new Map<string, { num: number; hit: Hit }>()
  for (const it of props.items) {
    const key = it.hit.doc_id || it.hit.doc_name || String(it.num)
    if (!seen.has(key)) seen.set(key, it)
  }
  return [...seen.values()]
})

const activePdfItem = computed(() => {
  if (props.highlightNum != null) {
    const highlighted = props.items.find((it) => it.num === props.highlightNum)
    if (highlighted) return highlighted
  }
  return props.items.find((it) => it.hit.doc_id) ?? docs.value[0] ?? null
})

const activePdfPage = computed(() => Math.max(1, Number(activePdfItem.value?.hit.page_start ?? 1) || 1))
const activePdfDocId = computed(() => activePdfItem.value?.hit.doc_id || activePdfItem.value?.hit.doc_name || '')
const activePdfCollection = computed(() => {
  const raw = activePdfItem.value?.hit.collection || activePdfItem.value?.hit.collection_name
  return typeof raw === 'string' ? raw : null
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

function scrollPdfToPage() {
  const scroll = registry.value?.getPlugin('scroll')?.provides?.()
  scroll?.scrollToPage({ pageNumber: activePdfPage.value })
}

function onPdfReady(r: PluginRegistry) {
  registry.value = r
  scrollPdfToPage()
}

async function loadPdfUrl() {
  if (props.tab !== 'pdf') return
  const docId = activePdfDocId.value
  if (!docId) {
    pdfUrl.value = ''
    pdfError.value = ''
    return
  }
  const key = `${activePdfCollection.value || ''}:${docId}`
  if (pdfDocKey.value === key && pdfUrl.value) {
    scrollPdfToPage()
    return
  }
  pdfLoading.value = true
  pdfError.value = ''
  try {
    const res = await api.getDocumentPdfUrl(docId, activePdfCollection.value)
    pdfDocKey.value = key
    pdfUrl.value = res.url
  } catch (e) {
    pdfUrl.value = ''
    pdfError.value = e instanceof Error ? e.message : String(e)
  } finally {
    pdfLoading.value = false
  }
}

watch(
  () => [props.tab, activePdfDocId.value, activePdfPage.value] as const,
  async () => {
    await loadPdfUrl()
    if (props.tab === 'pdf') scrollPdfToPage()
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
        <NButton
          :tertiary="tab === 'pdf'"
          :quaternary="tab !== 'pdf'"
          size="tiny"
          @click="emit('update:tab', 'pdf')"
        >
          {{ t('sources.original') }}
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
      <div v-else-if="tab === 'chunks'" class="flex flex-col gap-2.5">
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
            <NButton
              v-if="it.hit.doc_id"
              text
              size="tiny"
              class="!text-[10px]"
              @click="emit('update:tab', 'pdf')"
            >
              {{ t('sources.original') }}
            </NButton>
          </div>
          <p class="line-clamp-[12] whitespace-pre-wrap text-xs leading-relaxed text-muted">{{ it.hit.content }}</p>
        </div>
      </div>

      <!-- 原文 PDF -->
      <div v-else class="flex h-full min-h-0 flex-col">
        <div class="mb-2 flex items-center justify-between gap-2 text-xs text-muted">
          <div class="min-w-0">
            <div class="truncate text-base" :title="activePdfItem?.hit.doc_name">
              {{ activePdfItem?.hit.doc_name || activePdfDocId || t('common.empty') }}
            </div>
            <div v-if="activePdfDocId" class="text-faint">
              {{ t('sources.page', { n: activePdfPage }) }}
            </div>
          </div>
          <NButton size="tiny" quaternary :disabled="!pdfUrl" @click="scrollPdfToPage">
            {{ t('sources.jumpPage') }}
          </NButton>
        </div>
        <div class="min-h-0 flex-1 overflow-hidden rounded-[8px] border border-softer bg-surface-2">
          <div v-if="pdfLoading" class="grid h-full place-items-center text-xs text-faint">
            {{ t('common.loading') }}
          </div>
          <div v-else-if="pdfError" class="grid h-full place-items-center p-4 text-center text-xs text-red-500">
            {{ pdfError }}
          </div>
          <div v-else-if="!pdfUrl" class="grid h-full place-items-center text-xs text-faint">
            {{ t('sources.noPdf') }}
          </div>
          <PDFViewer
            v-else
            class="h-full w-full"
            :config="{ src: pdfUrl }"
            @ready="onPdfReady"
          />
        </div>
      </div>
    </div>
  </aside>
</template>
