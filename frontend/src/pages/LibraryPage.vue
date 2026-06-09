<script setup lang="ts">
import { computed, h, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NDataTable, NInput, type DataTableColumns } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import WorkspaceSplit from '@/components/WorkspaceSplit.vue'
import type { CollectionInfo, DocumentInfo, TaskResponse } from '@/api/types'

const { t } = useI18n()
const api = useApi()

const collections = ref<CollectionInfo[]>([])
const loading = ref(false)
const selected = ref<string | null>(null)
const docs = ref<DocumentInfo[]>([])
const docsLoading = ref(false)

const creating = ref(false)
const newName = ref('')

const uploading = ref(false)
const uploadProgress = ref('')
const fileInput = ref<HTMLInputElement | null>(null)

const selectedInfo = computed(() => collections.value.find((c) => c.name === selected.value) || null)
const canWriteSelected = computed(() => selectedInfo.value?.mine !== false)
const groupedCollections = computed(() => [
  {
    key: 'public',
    label: t('library.publicGroup'),
    items: collections.value.filter((c) => c.visibility === 'public' && !c.mine),
  },
  {
    key: 'org',
    label: t('library.orgGroup'),
    items: collections.value.filter((c) => c.visibility === 'org' && !c.mine),
  },
  {
    key: 'mine',
    label: t('library.mineGroup'),
    items: collections.value.filter((c) => c.mine || c.visibility === 'private'),
  },
])

async function loadCollections() {
  loading.value = true
  try {
    collections.value = (await api.listCollections()).collections
    if (!selected.value && collections.value.length) selected.value = collections.value[0].name
  } finally {
    loading.value = false
  }
}

async function loadDocs() {
  if (!selected.value) {
    docs.value = []
    return
  }
  docsLoading.value = true
  try {
    docs.value = await api.listDocuments(selected.value)
  } finally {
    docsLoading.value = false
  }
}

watch(selected, loadDocs)
onMounted(loadCollections)

async function createCollection() {
  const name = newName.value.trim()
  if (!name) return
  try {
    const c = await api.createCollection(name)
    newName.value = ''
    creating.value = false
    await loadCollections()
    selected.value = c.name
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  }
}

async function removeCollection(name: string) {
  if (!confirm(t('library.deleteCollectionConfirm'))) return
  await api.deleteCollection(name)
  if (selected.value === name) selected.value = null
  await loadCollections()
}

async function rebuild(name: string) {
  const task = await api.rebuildCollection(name)
  await pollTask(task, '重建')
  await loadCollections()
  await loadDocs()
}

async function toggleVisibility(c: CollectionInfo) {
  const next = c.visibility === 'private' ? 'org' : c.visibility === 'org' ? 'public' : 'private'
  try {
    await api.setCollectionVisibility(c.name, next)
    await loadCollections()
  } catch {
    alert('后端暂未支持可见性设置（待 M5）')
  }
}

async function copyToMine(c: CollectionInfo) {
  try {
    const copied = await api.copyCollectionToMine(c.name)
    await loadCollections()
    selected.value = copied.id
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  }
}

function visibilityLabel(v?: string): string {
  if (v === 'public') return t('common.public')
  if (v === 'org') return t('common.org')
  return t('common.private')
}

async function pollTask(task: TaskResponse, label: string): Promise<void> {
  let cur = task
  for (let i = 0; i < 600 && cur.status !== 'done' && cur.status !== 'failed'; i++) {
    uploadProgress.value = `${label} ${Math.round((cur.progress || 0) * 100)}%`
    await new Promise((r) => setTimeout(r, 1500))
    try {
      cur = await api.getTask(task.id)
    } catch {
      break
    }
  }
  uploadProgress.value = cur.status === 'failed' ? `${label}失败` : ''
}

async function onFiles(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files ?? [])
  if (files.length === 0 || !selected.value) return
  uploading.value = true
  try {
    const task = await api.uploadAndIngest(files, selected.value)
    await pollTask(task, t('library.parsing'))
    await loadCollections()
    await loadDocs()
  } catch (err) {
    alert(err instanceof Error ? err.message : String(err))
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

async function removeDoc(docId: string) {
  if (!selected.value || !confirm(t('library.deleteDocConfirm'))) return
  try {
    await api.deleteDocument(selected.value, docId)
    await loadDocs()
  } catch {
    alert('后端暂未支持单文献删除（待 M5）')
  }
}

function statusLabel(s?: string): string {
  return s === 'ready' ? t('library.ready') : s === 'failed' ? t('library.failed') : t('library.parsing')
}

const docColumns = computed<DataTableColumns<DocumentInfo>>(() => [
  {
    title: t('common.name'),
    key: 'title',
    ellipsis: { tooltip: true },
    render: (row) => row.title || row.doc_id,
  },
  {
    title: t('common.status'),
    key: 'status',
    width: 96,
    render: (row) => h('span', { class: 'chip' }, statusLabel(row.status)),
  },
  {
    title: t('sources.chunks'),
    key: 'chunk_count',
    width: 88,
    render: (row) => row.chunk_count ?? '-',
  },
  {
    title: '',
    key: 'actions',
    width: 48,
    align: 'right',
    render: (row) =>
      h(
        NButton,
        {
          quaternary: true,
          circle: true,
          size: 'small',
          title: t('common.delete'),
          onClick: () => removeDoc(row.doc_id),
        },
        { icon: () => h('span', { class: 'i-lucide-trash-2 text-sm text-red-500' }) },
      ),
  },
])
</script>

<template>
  <WorkspaceSplit>
    <!-- 文献库列表 -->
    <template #pane1>
    <div class="workspace-card flex flex-col">
      <header class="flex items-center justify-between border-b border-softer px-4 py-3">
        <span class="text-sm font-semibold text-base">{{ t('library.collections') }}</span>
        <div class="flex items-center gap-1.5">
          <NButton quaternary circle size="small" :title="t('common.refresh')" @click="loadCollections">
            <template #icon>
              <span class="i-lucide-rotate-cw text-sm" />
            </template>
          </NButton>
          <NButton quaternary circle size="small" :title="t('library.newCollection')" @click="creating = !creating">
            <template #icon>
              <span class="i-lucide-plus" />
            </template>
          </NButton>
        </div>
      </header>

      <div v-if="creating" class="border-b border-softer p-2">
        <NInput
          v-model:value="newName"
          size="small"
          :placeholder="t('library.collectionName')"
          @keydown.enter="createCollection"
        />
        <div class="mt-1.5 flex justify-end gap-1.5">
          <NButton quaternary size="tiny" @click="creating = false">{{ t('common.cancel') }}</NButton>
          <NButton type="primary" size="tiny" @click="createCollection">{{ t('common.create') }}</NButton>
        </div>
      </div>

      <div class="min-h-0 flex-1 overflow-y-auto p-2">
        <p v-if="loading" class="px-2 py-4 text-center text-xs text-faint">{{ t('common.loading') }}</p>
        <p v-else-if="collections.length === 0" class="px-2 py-6 text-center text-xs text-faint">
          {{ t('library.emptyCollections') }}
        </p>
        <template v-for="group in groupedCollections" :key="group.key">
          <div v-if="group.items.length" class="px-2 pb-1 pt-2 text-[10px] font-medium uppercase tracking-wide text-faint">
            {{ group.label }}
          </div>
          <div
            v-for="c in group.items"
            :key="c.name"
            class="group mb-0.5 flex cursor-pointer items-center gap-2 rounded-[6px] px-2 py-2 transition-colors"
            :class="selected === c.name ? 'bg-active hover:bg-active-hover' : 'hover:bg-hover'"
            @click="selected = c.name"
          >
            <span class="i-lucide-folder shrink-0 text-muted" />
            <div class="min-w-0 flex-1">
              <div class="truncate text-sm text-base" :title="c.display_name || c.name">
                {{ c.display_name || c.name.replace(/^kb_/, '') }}
              </div>
              <div class="flex items-center gap-1.5 text-[10px] text-faint">
                <span>{{ t('library.docCount', { n: c.doc_count ?? 0 }) }}</span>
                <span>·</span>
                <span>{{ t('library.rowCount', { n: c.row_count }) }}</span>
                <span class="chip !px-1 !py-0 text-accent">{{ visibilityLabel(c.visibility) }}</span>
              </div>
            </div>
            <NButton
              v-if="!c.mine"
              class="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
              quaternary
              circle
              size="tiny"
              :title="t('common.copyToMine')"
              @click.stop="copyToMine(c)"
            >
              <template #icon>
                <span class="i-lucide-copy-plus text-xs" />
              </template>
            </NButton>
            <NButton
              v-else
              class="shrink-0 opacity-0 transition-opacity hover:text-red-500 group-hover:opacity-100"
              quaternary
              circle
              size="tiny"
              :title="t('common.delete')"
              @click.stop="removeCollection(c.name)"
            >
              <template #icon>
                <span class="i-lucide-trash-2 text-xs text-red-500" />
              </template>
            </NButton>
          </div>
        </template>
      </div>
    </div>
    </template>

    <!-- 文献列表 -->
    <template #pane2>
    <div class="workspace-card flex flex-col">
      <header class="flex items-center justify-between border-b border-softer px-5 py-3.5">
        <div class="min-w-0">
          <div class="truncate text-sm font-semibold text-base">
            {{ selectedInfo ? (selectedInfo.display_name || selectedInfo.name) : t('library.title') }}
          </div>
          <div v-if="uploadProgress" class="text-xs text-accent">{{ uploadProgress }}</div>
        </div>
        <div v-if="selected" class="flex items-center gap-2">
          <NButton v-if="canWriteSelected" tertiary size="small" :title="t('library.rebuild')" @click="rebuild(selected)">
            <template #icon>
              <span class="i-lucide-hammer" />
            </template>
            {{ t('library.rebuild') }}
          </NButton>
          <NButton
            v-if="selectedInfo"
            :disabled="!canWriteSelected"
            tertiary
            size="small"
            @click="toggleVisibility(selectedInfo)"
          >
            <template #icon>
              <span :class="selectedInfo.visibility === 'org' ? 'i-lucide-lock' : 'i-lucide-users'" />
            </template>
            {{ visibilityLabel(selectedInfo.visibility) }}
          </NButton>
          <NButton v-if="canWriteSelected" type="primary" size="small" :loading="uploading" :disabled="uploading" @click="fileInput?.click()">
            <template #icon>
              <span class="i-lucide-upload" />
            </template>
            {{ t('library.uploadDocs') }}
          </NButton>
          <input ref="fileInput" type="file" accept="application/pdf" multiple class="hidden" @change="onFiles" />
        </div>
      </header>

      <div class="min-h-0 flex-1 overflow-y-auto p-5">
        <div v-if="!selected" class="grid h-full place-items-center text-sm text-faint">
          {{ t('library.emptyCollections') }}
        </div>
        <div v-else-if="docsLoading" class="text-center text-xs text-faint">{{ t('common.loading') }}</div>
        <div v-else-if="docs.length === 0" class="grid h-full place-items-center text-center text-sm text-faint">
          <div>
            <p>{{ t('library.emptyDocs') }}</p>
            <p class="mt-1 text-xs">{{ t('library.uploadHint') }}</p>
          </div>
        </div>
        <NDataTable
          v-else
          :columns="docColumns"
          :data="docs"
          :row-key="(row) => row.doc_id"
          size="small"
          :bordered="false"
        />
      </div>
    </div>
    </template>
  </WorkspaceSplit>
</template>
