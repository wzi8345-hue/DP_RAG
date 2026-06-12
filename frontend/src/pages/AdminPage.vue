<script setup lang="ts">
import { computed, h, onMounted, shallowRef } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NDataTable, NTabPane, NTabs, type DataTableColumns } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import { useAuthz } from '@/composables/useAuthz'
import type {
  AdminAuditLog,
  AdminGenerationRun,
  AdminIngestTask,
  BackendConversationPayload,
  CollectionInfo,
  SkillSummary,
} from '@/api/types'

const { t } = useI18n()
const api = useApi()
const authz = useAuthz()

const loading = shallowRef(false)
const collections = shallowRef<CollectionInfo[]>([])
const conversations = shallowRef<BackendConversationPayload[]>([])
const skills = shallowRef<SkillSummary[]>([])
const ingestTasks = shallowRef<AdminIngestTask[]>([])
const generationRuns = shallowRef<AdminGenerationRun[]>([])
const auditLogs = shallowRef<AdminAuditLog[]>([])

function terminal(status?: string) {
  return status === 'done' || status === 'failed' || status === 'cancelled' || status === 'stopped'
}

function visibilityLabel(value?: string) {
  if (value === 'public') return t('common.public')
  if (value === 'org') return t('common.org')
  return t('common.private')
}

function formatTime(value?: string | null) {
  if (!value) return '-'
  return new Date(value).toLocaleString()
}

async function load() {
  if (!authz.isAdmin.value) return
  loading.value = true
  try {
    const [kb, conv, skill, ingest, runs, logs] = await Promise.all([
      api.adminCollections(),
      api.adminConversations(),
      api.adminSkills(),
      api.adminIngestTasks(),
      api.adminGenerationRuns(),
      api.adminAuditLogs(),
    ])
    collections.value = kb.collections ?? []
    conversations.value = conv.conversations ?? []
    skills.value = skill.skills ?? []
    ingestTasks.value = ingest.tasks ?? []
    generationRuns.value = runs.runs ?? []
    auditLogs.value = logs.logs ?? []
  } finally {
    loading.value = false
  }
}

async function removeCollection(row: CollectionInfo) {
  if (!confirm(t('library.deleteCollectionConfirm'))) return
  await api.deleteCollection(row.name)
  await load()
}

async function removeSkill(row: SkillSummary) {
  if (!confirm(t('skills.deleteConfirm'))) return
  await api.deleteSkill(row.id)
  await load()
}

async function cancelIngest(row: AdminIngestTask) {
  await api.cancelIngestTask(row.id)
  await load()
}

async function stopRun(row: AdminGenerationRun) {
  await api.stopRun(row.id)
  await load()
}

const collectionColumns = computed<DataTableColumns<CollectionInfo>>(() => [
  { title: t('common.name'), key: 'display_name', render: (row) => row.display_name || row.name },
  { title: 'Owner', key: 'owner_id', ellipsis: { tooltip: true } },
  { title: 'Org', key: 'org_id', ellipsis: { tooltip: true } },
  { title: t('common.visibility'), key: 'visibility', render: (row) => visibilityLabel(row.visibility) },
  { title: t('library.documents'), key: 'doc_count', render: (row) => row.doc_count ?? 0 },
  {
    title: '',
    key: 'actions',
    align: 'right',
    width: 72,
    render: (row) =>
      h(
        NButton,
        { quaternary: true, size: 'small', type: 'error', onClick: () => removeCollection(row) },
        { default: () => t('common.delete') },
      ),
  },
])

const conversationColumns = computed<DataTableColumns<BackendConversationPayload>>(() => [
  { title: t('chat.title'), key: 'title', render: (row) => row.title || row.id },
  { title: 'Owner', key: 'ownerId', ellipsis: { tooltip: true } },
  { title: 'Org', key: 'orgId', ellipsis: { tooltip: true } },
  { title: t('common.visibility'), key: 'visibility', render: (row) => visibilityLabel(row.visibility) },
  { title: 'Updated', key: 'updatedAt', render: (row) => (row.updatedAt ? new Date(row.updatedAt).toLocaleString() : '-') },
])

const skillColumns = computed<DataTableColumns<SkillSummary>>(() => [
  { title: t('common.name'), key: 'name', render: (row) => row.name || row.id },
  { title: 'Owner', key: 'owner_id', ellipsis: { tooltip: true } },
  { title: 'Org', key: 'org_id', ellipsis: { tooltip: true } },
  { title: t('common.visibility'), key: 'visibility', render: (row) => visibilityLabel(row.visibility) },
  {
    title: '',
    key: 'actions',
    align: 'right',
    width: 72,
    render: (row) =>
      h(
        NButton,
        { quaternary: true, size: 'small', type: 'error', onClick: () => removeSkill(row) },
        { default: () => t('common.delete') },
      ),
  },
])

const ingestColumns = computed<DataTableColumns<AdminIngestTask>>(() => [
  { title: 'Task', key: 'id', ellipsis: { tooltip: true } },
  { title: t('library.collections'), key: 'collection_name', ellipsis: { tooltip: true } },
  { title: 'Owner', key: 'owner_id', ellipsis: { tooltip: true } },
  { title: t('common.status'), key: 'status' },
  { title: 'Progress', key: 'progress', render: (row) => `${Math.round((row.progress || 0) * 100)}%` },
  {
    title: '',
    key: 'actions',
    align: 'right',
    width: 80,
    render: (row) =>
      terminal(row.status)
        ? null
        : h(NButton, { quaternary: true, size: 'small', onClick: () => cancelIngest(row) }, { default: () => t('common.cancel') }),
  },
])

const runColumns = computed<DataTableColumns<AdminGenerationRun>>(() => [
  { title: 'Run', key: 'id', ellipsis: { tooltip: true } },
  { title: t('chat.title'), key: 'conversation_id', ellipsis: { tooltip: true } },
  { title: 'Owner', key: 'owner_id', ellipsis: { tooltip: true } },
  { title: t('common.status'), key: 'status' },
  { title: 'Created', key: 'created_at', render: (row) => formatTime(row.created_at) },
  {
    title: '',
    key: 'actions',
    align: 'right',
    width: 80,
    render: (row) =>
      terminal(row.status)
        ? null
        : h(NButton, { quaternary: true, size: 'small', onClick: () => stopRun(row) }, { default: () => t('chat.stop') }),
  },
])

const auditColumns = computed<DataTableColumns<AdminAuditLog>>(() => [
  { title: 'Action', key: 'action' },
  { title: 'Resource', key: 'resource_id', ellipsis: { tooltip: true } },
  { title: 'Type', key: 'resource_type' },
  { title: 'Actor', key: 'actor_id', ellipsis: { tooltip: true } },
  { title: 'Target', key: 'target_owner_id', ellipsis: { tooltip: true } },
  { title: 'Created', key: 'created_at', render: (row) => formatTime(row.created_at) },
])

onMounted(load)
</script>

<template>
  <div class="workspace-card flex h-full flex-col">
    <header class="flex items-center justify-between border-b border-softer px-5 py-3.5">
      <div>
        <h1 class="text-sm font-semibold text-base">{{ t('admin.title') }}</h1>
        <p class="text-xs text-faint">{{ authz.role.value }} · {{ authz.orgId.value || t('admin.noOrg') }}</p>
      </div>
      <NButton quaternary size="small" :loading="loading" @click="load">
        <template #icon>
          <span class="i-lucide-rotate-cw" />
        </template>
        {{ t('common.refresh') }}
      </NButton>
    </header>

    <div v-if="!authz.isAdmin.value" class="grid flex-1 place-items-center text-sm text-faint">
      {{ t('admin.forbidden') }}
    </div>

    <NTabs v-else class="min-h-0 flex-1 px-4 py-3" pane-class="min-h-0">
      <NTabPane name="collections" :tab="t('library.collections')">
        <NDataTable :columns="collectionColumns" :data="collections" size="small" :bordered="false" />
      </NTabPane>
      <NTabPane name="conversations" :tab="t('chat.title')">
        <NDataTable :columns="conversationColumns" :data="conversations" size="small" :bordered="false" />
      </NTabPane>
      <NTabPane name="skills" :tab="t('skills.title')">
        <NDataTable :columns="skillColumns" :data="skills" size="small" :bordered="false" />
      </NTabPane>
      <NTabPane name="ingest" :tab="t('admin.ingestTasks')">
        <NDataTable :columns="ingestColumns" :data="ingestTasks" size="small" :bordered="false" />
      </NTabPane>
      <NTabPane name="runs" :tab="t('admin.generationRuns')">
        <NDataTable :columns="runColumns" :data="generationRuns" size="small" :bordered="false" />
      </NTabPane>
      <NTabPane name="audit" :tab="t('admin.auditLogs')">
        <NDataTable :columns="auditColumns" :data="auditLogs" size="small" :bordered="false" />
      </NTabPane>
    </NTabs>
  </div>
</template>
