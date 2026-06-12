<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NInput, NInputNumber } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import { useAuthz } from '@/composables/useAuthz'
import WorkspaceSplit from '@/components/WorkspaceSplit.vue'
import type { SkillSpec, SkillSummary, Visibility } from '@/api/types'

interface SkillForm {
  id: string
  name: string
  description: string
  priority: number
  triggersText: string
  plan: string
  policy: string
  synthesis_system: string
  synthesis_user: string
}

const { t } = useI18n()
const api = useApi()
const authz = useAuthz()

const enabled = ref(true)
const skills = ref<SkillSummary[]>([])
const loading = ref(false)
const selectedId = ref<string | null>(null)
const editing = ref(false)
const saving = ref(false)

const form = reactive<SkillForm>({
  id: '',
  name: '',
  description: '',
  priority: 50,
  triggersText: '',
  plan: '',
  policy: '',
  synthesis_system: '',
  synthesis_user: '',
})

const selected = computed(() => skills.value.find((s) => s.id === selectedId.value) || null)
const canManageSelected = computed(() => Boolean(selected.value?.can_manage ?? selected.value?.mine))
const groupedSkills = computed(() => [
  {
    key: 'public',
    label: t('common.public'),
    items: skills.value.filter((s) => s.visibility === 'public' && !s.mine),
  },
  {
    key: 'org',
    label: t('common.org'),
    items: skills.value.filter((s) => s.visibility === 'org' && !s.mine),
  },
  {
    key: 'mine',
    label: t('common.mine'),
    items: skills.value.filter((s) => s.mine || s.visibility === 'private'),
  },
])

async function load() {
  loading.value = true
  try {
    const r = await api.listSkills()
    enabled.value = r.enabled
    skills.value = r.skills
    if (!selectedId.value && skills.value.length) selectedId.value = skills.value[0].id
  } finally {
    loading.value = false
  }
}
onMounted(load)

function startNew() {
  Object.assign(form, {
    id: '',
    name: '',
    description: '',
    priority: 50,
    triggersText: '',
    plan: '',
    policy: '',
    synthesis_system: '',
    synthesis_user: '',
  })
  selectedId.value = null
  editing.value = true
}

function startEdit(s: SkillSummary) {
  Object.assign(form, {
    id: s.id,
    name: s.name,
    description: s.description ?? '',
    priority: s.priority ?? 50,
    triggersText: (s.triggers ?? []).join(', '),
    plan: s.plan ?? '',
    policy: s.policy ?? '',
    synthesis_system: s.synthesis_system ?? '',
    synthesis_user: s.synthesis_user ?? '',
  })
  editing.value = true
}

async function save() {
  if (!form.id.trim() || !form.name.trim()) return
  saving.value = true
  try {
    const spec: SkillSpec = {
      id: form.id.trim(),
      name: form.name.trim(),
      description: form.description,
      priority: form.priority,
      triggers: form.triggersText.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
      plan: form.plan,
      policy: form.policy,
      synthesis_system: form.synthesis_system,
      synthesis_user: form.synthesis_user,
    }
    await api.saveSkill(spec)
    editing.value = false
    await load()
    selectedId.value = spec.id
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  } finally {
    saving.value = false
  }
}

async function remove(s: SkillSummary) {
  if (!confirm(t('skills.deleteConfirm'))) return
  try {
    await api.deleteSkill(s.id)
    if (selectedId.value === s.id) selectedId.value = null
    await load()
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  }
}

async function toggleVisibility(s: SkillSummary) {
  try {
    const next: Visibility =
      s.visibility === 'private'
        ? authz.canUseOrgVisibility.value ? 'org' : 'public'
        : s.visibility === 'org' ? 'public' : 'private'
    await api.setSkillVisibility(s.id, next)
    await load()
  } catch {
    alert('后端暂未支持可见性设置（待 M5）')
  }
}

async function copyToMine(s: SkillSummary) {
  try {
    const copied = await api.copySkillToMine(s.id)
    await load()
    selectedId.value = copied.id
    editing.value = false
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  }
}

function visibilityLabel(v?: string): string {
  if (v === 'public') return t('common.public')
  if (v === 'org') return t('common.org')
  return t('common.private')
}
</script>

<template>
  <WorkspaceSplit>
    <template #pane1>
    <div class="workspace-card flex flex-col">
      <header class="flex items-center justify-between border-b border-softer px-4 py-3">
        <span class="text-sm font-semibold text-base">{{ t('skills.title') }}</span>
        <NButton quaternary circle size="small" :title="t('skills.new')" @click="startNew">
          <template #icon>
            <span class="i-lucide-plus" />
          </template>
        </NButton>
      </header>
      <div v-if="!enabled" class="m-2 rounded-[8px] bg-surface-2 p-3 text-xs text-faint">
        {{ t('skills.disabled') }}
      </div>
      <div class="min-h-0 flex-1 overflow-y-auto p-2">
        <p v-if="loading" class="px-2 py-4 text-center text-xs text-faint">{{ t('common.loading') }}</p>
        <p v-else-if="skills.length === 0" class="px-2 py-6 text-center text-xs text-faint">{{ t('skills.empty') }}</p>
        <template v-for="group in groupedSkills" :key="group.key">
          <div v-if="group.items.length" class="px-2 pb-1 pt-2 text-[10px] font-medium uppercase tracking-wide text-faint">
            {{ group.label }}
          </div>
          <div
            v-for="s in group.items"
            :key="s.id"
            class="group mb-0.5 flex cursor-pointer items-center gap-2 rounded-[6px] px-2 py-2 transition-colors"
            :class="selectedId === s.id && !editing ? 'bg-active hover:bg-active-hover' : 'hover:bg-hover'"
            @click="selectedId = s.id; editing = false"
          >
            <span class="i-lucide-puzzle shrink-0 text-muted" />
            <div class="min-w-0 flex-1">
              <div class="truncate text-sm text-base">{{ s.name }}</div>
              <div class="flex items-center gap-1 text-[10px] text-faint">
                <span class="chip !px-1 !py-0">{{ s.editable ? t('skills.editable') : t('skills.builtin') }}</span>
                <span>P{{ s.priority }}</span>
                <span class="chip !px-1 !py-0 text-accent">{{ visibilityLabel(s.visibility) }}</span>
              </div>
            </div>
            <NButton
              v-if="!s.mine"
              class="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
              quaternary
              circle
              size="tiny"
              :title="t('common.copyToMine')"
              @click.stop="copyToMine(s)"
            >
              <template #icon>
                <span class="i-lucide-copy-plus text-xs" />
              </template>
            </NButton>
          </div>
        </template>
      </div>
    </div>
    </template>

    <template #pane2>
    <div class="workspace-card">
      <div class="h-full overflow-y-auto p-5">
      <!-- 编辑表单 -->
      <div v-if="editing" class="mx-auto max-w-2xl">
        <h2 class="mb-4 text-base font-semibold">{{ form.id ? t('common.edit') : t('skills.new') }}</h2>
        <div class="flex flex-col gap-3">
          <div class="grid grid-cols-2 gap-3">
            <label class="flex flex-col gap-1">
              <span class="field-label">{{ t('skills.fields.id') }}</span>
              <NInput v-model:value="form.id" size="small" :disabled="!!selected" />
            </label>
            <label class="flex flex-col gap-1">
              <span class="field-label">{{ t('skills.fields.name') }}</span>
              <NInput v-model:value="form.name" size="small" />
            </label>
          </div>
          <label class="flex flex-col gap-1">
            <span class="field-label">{{ t('skills.fields.description') }}</span>
            <NInput v-model:value="form.description" size="small" />
          </label>
          <div class="grid grid-cols-2 gap-3">
            <label class="flex flex-col gap-1">
              <span class="field-label">{{ t('skills.fields.priority') }}</span>
              <NInputNumber v-model:value="form.priority" size="small" />
            </label>
            <label class="flex flex-col gap-1">
              <span class="field-label">{{ t('skills.fields.triggers') }}</span>
              <NInput v-model:value="form.triggersText" size="small" />
            </label>
          </div>
          <label class="flex flex-col gap-1">
            <span class="field-label">{{ t('skills.fields.plan') }}</span>
            <NInput v-model:value="form.plan" type="textarea" :rows="3" />
          </label>
          <label class="flex flex-col gap-1">
            <span class="field-label">{{ t('skills.fields.policy') }}</span>
            <NInput v-model:value="form.policy" type="textarea" :rows="3" />
          </label>
          <label class="flex flex-col gap-1">
            <span class="field-label">{{ t('skills.fields.synthesisSystem') }}</span>
            <NInput v-model:value="form.synthesis_system" type="textarea" :rows="3" />
          </label>
          <label class="flex flex-col gap-1">
            <span class="field-label">{{ t('skills.fields.synthesisUser') }}</span>
            <NInput v-model:value="form.synthesis_user" type="textarea" :rows="3" />
          </label>
          <div class="flex justify-end gap-2">
            <NButton quaternary @click="editing = false">{{ t('common.cancel') }}</NButton>
            <NButton type="primary" :loading="saving" :disabled="saving" @click="save">
              {{ t('common.save') }}
            </NButton>
          </div>
        </div>
      </div>

      <!-- 详情 -->
      <div v-else-if="selected" class="mx-auto max-w-2xl">
        <div class="mb-3 flex items-start justify-between">
          <div>
            <h2 class="text-base font-semibold">{{ selected.name }}</h2>
            <p class="mt-0.5 text-sm text-muted">{{ selected.description || '—' }}</p>
          </div>
          <div class="flex items-center gap-2">
            <template v-if="selected.editable && canManageSelected">
              <NButton tertiary size="small" @click="toggleVisibility(selected)">
                {{ visibilityLabel(selected.visibility) }}
              </NButton>
              <NButton tertiary size="small" @click="startEdit(selected)">
                <template #icon>
                  <span class="i-lucide-pencil" />
                </template>
                {{ t('common.edit') }}
              </NButton>
              <NButton type="error" size="small" @click="remove(selected)">
                <template #icon>
                  <span class="i-lucide-trash-2" />
                </template>
                {{ t('common.delete') }}
              </NButton>
            </template>
            <NButton v-else-if="selected.mine === false" tertiary size="small" @click="copyToMine(selected)">
              {{ t('common.copyToMine') }}
            </NButton>
            <span v-else class="chip">{{ t('skills.builtin') }}</span>
          </div>
        </div>
        <dl class="flex flex-col gap-3 text-sm">
          <div>
            <dt class="field-label">{{ t('skills.fields.priority') }}</dt>
            <dd class="text-base">{{ selected.priority }}</dd>
          </div>
          <div v-if="selected.triggers?.length">
            <dt class="field-label">{{ t('skills.fields.triggers') }}</dt>
            <dd class="mt-1 flex flex-wrap gap-1">
              <span v-for="tr in selected.triggers" :key="tr" class="chip">{{ tr }}</span>
            </dd>
          </div>
          <div v-if="selected.plan">
            <dt class="field-label">{{ t('skills.fields.plan') }}</dt>
            <dd class="mt-1 whitespace-pre-wrap rounded-[8px] bg-surface-2 p-3 font-mono text-xs text-muted">{{ selected.plan }}</dd>
          </div>
          <div v-if="selected.synthesis_system">
            <dt class="field-label">{{ t('skills.fields.synthesisSystem') }}</dt>
            <dd class="mt-1 whitespace-pre-wrap rounded-[8px] bg-surface-2 p-3 font-mono text-xs text-muted">{{ selected.synthesis_system }}</dd>
          </div>
        </dl>
      </div>

      <div v-else class="grid h-full place-items-center text-sm text-faint">
        {{ t('skills.empty') }}
      </div>
      </div>
    </div>
    </template>
  </WorkspaceSplit>
</template>
