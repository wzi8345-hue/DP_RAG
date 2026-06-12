<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NCheckbox, NPopover, NSelect } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import type { ComposerState } from '@/composables/useChat'
import type { CollectionInfo, DocumentInfo } from '@/api/types'

const props = defineProps<{ composer: ComposerState; busy: boolean }>()
const emit = defineEmits<{ (e: 'send', query: string): void; (e: 'stop'): void }>()

const { t } = useI18n()
const api = useApi()

const input = ref('')
const collections = ref<CollectionInfo[]>([])
const docs = ref<DocumentInfo[]>([])
const uploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const collectionOptions = computed(() => [
  { label: t('composer.allDocs'), value: '' },
  ...collections.value.map((c) => ({
    label: `${c.display_name || c.name.replace(/^kb_/, '')} (${c.row_count})`,
    value: c.name,
  })),
])

async function loadCollections() {
  try {
    collections.value = (await api.listCollections()).collections
  } catch {
    collections.value = []
  }
}

async function loadDocs() {
  props.composer.docIds = []
  if (!props.composer.collection) {
    docs.value = []
    return
  }
  docs.value = await api.listDocuments(props.composer.collection)
}

watch(() => props.composer.collection, loadDocs)
onMounted(loadCollections)

function submit() {
  const q = input.value.trim()
  if (!q || props.busy) return
  emit('send', q)
  input.value = ''
}

function toggleDoc(docId: string) {
  const i = props.composer.docIds.indexOf(docId)
  if (i >= 0) props.composer.docIds.splice(i, 1)
  else props.composer.docIds.push(docId)
}

async function onFiles(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files ?? [])
  if (files.length === 0) return
  uploading.value = true
  try {
    const target = props.composer.collection || '我的上传'
    await api.uploadAndIngest(files, target)
    await loadCollections()
    if (!props.composer.collection) {
      const found = collections.value.find((c) => c.display_name === '我的上传' || c.name.includes('uploads'))
      if (found) props.composer.collection = found.name
    }
    await loadDocs()
  } catch {
    /* surfaced elsewhere */
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}
</script>

<template>
  <div class="border-t border-softer px-6 py-4">
    <div class="mx-auto max-w-3xl">
      <!-- 控制行 -->
      <div class="mb-2.5 flex flex-wrap items-center gap-2 text-xs">
        <!-- 模式 -->
        <div class="flex items-center rounded-[6px] bg-surface p-0.5">
          <NButton
            :tertiary="!composer.professional"
            :quaternary="composer.professional"
            size="tiny"
            :disabled="busy"
            :title="t('composer.quickHint')"
            @click="composer.professional = false"
          >
            {{ t('composer.quickMode') }}
          </NButton>
          <NButton
            :tertiary="composer.professional"
            :quaternary="!composer.professional"
            size="tiny"
            :disabled="busy"
            :title="t('composer.expertHint')"
            @click="composer.professional = true"
          >
            {{ t('composer.expertMode') }}
          </NButton>
        </div>

        <!-- 文献库 -->
        <NSelect
          v-model:value="composer.collection"
          :options="collectionOptions"
          size="small"
          class="w-40"
          :disabled="busy"
          :title="t('composer.library')"
        />

        <!-- 选择文献 -->
        <NPopover
          trigger="click"
          placement="top-start"
          :disabled="busy || !composer.collection || docs.length === 0"
          style="padding: 4px"
        >
          <template #trigger>
            <NButton
              :tertiary="composer.docIds.length > 0"
              :quaternary="composer.docIds.length === 0"
              size="tiny"
              :disabled="busy || !composer.collection || docs.length === 0"
              :title="t('composer.selectDocs')"
            >
              <template #icon>
                <span class="i-lucide-file-text" />
              </template>
              {{ composer.docIds.length ? t('composer.selectedDocs', { n: composer.docIds.length }) : t('composer.selectDocs') }}
            </NButton>
          </template>
          <div class="max-h-60 w-60 overflow-y-auto">
            <label
              v-for="d in docs"
              :key="d.doc_id"
              class="flex cursor-pointer items-center gap-2 rounded-[6px] px-2 py-1.5 text-xs hover:bg-hover"
            >
              <NCheckbox
                :checked="composer.docIds.includes(d.doc_id)"
                @update:checked="toggleDoc(d.doc_id)"
              />
              <span class="truncate">{{ d.title || d.doc_id }}</span>
            </label>
          </div>
        </NPopover>

        <!-- 开关 -->
        <NButton
          :tertiary="composer.enableRetrieval"
          :quaternary="!composer.enableRetrieval"
          size="tiny"
          :disabled="busy"
          :title="t('composer.enableRetrieval')"
          @click="composer.enableRetrieval = !composer.enableRetrieval"
        >
          <template #icon>
            <span class="i-lucide-search" />
          </template>
          {{ t('composer.enableRetrieval') }}
        </NButton>
        <NButton
          :tertiary="composer.useAgentic"
          :quaternary="!composer.useAgentic"
          size="tiny"
          :disabled="busy"
          title="Agentic"
          @click="composer.useAgentic = !composer.useAgentic"
        >
          <template #icon>
            <span class="i-lucide-sparkles" />
          </template>
          {{ t('composer.agentic') }}
        </NButton>

        <!-- 上传 -->
        <NButton quaternary size="tiny" :loading="uploading" :disabled="busy || uploading" :title="t('composer.attach')" @click="fileInput?.click()">
          <template #icon>
            <span class="i-lucide-upload" />
          </template>
          {{ t('composer.attach') }}
        </NButton>
        <input ref="fileInput" type="file" accept="application/pdf" multiple class="hidden" @change="onFiles" />
      </div>

      <!-- 输入 -->
      <div class="flex items-end gap-2">
        <textarea
          v-model="input"
          rows="1"
          :placeholder="t('chat.placeholder')"
          class="input max-h-40 min-h-[2.5rem] flex-1 resize-none py-2.5"
          @keydown.enter.exact.prevent="submit"
        />
        <NButton v-if="busy" type="error" class="h-10 px-4" @click="emit('stop')">
          <template #icon>
            <span class="i-lucide-square" />
          </template>
          {{ t('chat.stop') }}
        </NButton>
        <NButton v-else type="primary" class="h-10 px-4" :disabled="!input.trim()" @click="submit">
          <template #icon>
            <span class="i-lucide-send-horizontal" />
          </template>
          {{ t('chat.send') }}
        </NButton>
      </div>
      <p v-if="busy" class="mt-1.5 text-center text-xs text-faint">{{ t('chat.busyHint') }}</p>
    </div>
  </div>
</template>
