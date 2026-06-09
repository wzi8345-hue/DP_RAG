<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'
import { useConversationsStore } from '@/stores/conversations'
import { parseCitations } from '@/utils/citations'
import { wrapBareLatex } from '@/utils/latex'
import type { ChatMessage, Conversation } from '@/api/types'
import MarkdownMessage from './MarkdownMessage.vue'
import BranchNav from './BranchNav.vue'
import ResearchBadges from './ResearchBadges.vue'

const props = defineProps<{ msg: ChatMessage; conv: Conversation; busy: boolean }>()
const emit = defineEmits<{
  (e: 'sources', msgId: string): void
  (e: 'cite', msgId: string, num: number): void
  (e: 'regenerate', assistantId: string): void
  (e: 'edit', userId: string, newContent: string): void
}>()

const { t } = useI18n()
const store = useConversationsStore()

const parsed = computed(() =>
  parseCitations(props.msg.content, props.msg.hits ?? [], props.msg.context, {
    research: props.msg.expert,
  }),
)
const isStreaming = computed(() => props.msg.status === 'streaming')
const rendered = computed(() =>
  wrapBareLatex(
    isStreaming.value
      ? props.msg.content // streaming：markstream 处理增量；citations 完成后再解析
      : parsed.value.markdown,
  ),
)
const sourceCount = computed(() =>
  parsed.value.hasCitations ? parsed.value.citedHits.length : props.msg.hits?.length ?? 0,
)

const branch = computed(() => store.branchInfo(props.conv, props.msg.id))
const thinkingOpen = ref(true)

function onCite(num: number) {
  emit('cite', props.msg.id, num)
}

function prevBranch() {
  const b = branch.value
  if (b.index > 0) store.switchBranch(props.conv, b.siblings[b.index - 1].id)
}
function nextBranch() {
  const b = branch.value
  if (b.index < b.total - 1) store.switchBranch(props.conv, b.siblings[b.index + 1].id)
}

// 编辑历史 user 消息
const editing = ref(false)
const draft = ref('')
const editRef = ref<HTMLTextAreaElement | null>(null)
async function startEdit() {
  draft.value = props.msg.content
  editing.value = true
  await nextTick()
  editRef.value?.focus()
}
function submitEdit() {
  if (!draft.value.trim()) return
  emit('edit', props.msg.id, draft.value)
  editing.value = false
}
</script>

<template>
  <!-- 用户消息 -->
  <div v-if="msg.role === 'user'" class="group flex justify-end">
    <div class="max-w-[80%]">
      <div v-if="!editing" class="rounded-[8px] rounded-br-[3px] bg-accent px-3.5 py-2 text-sm text-white whitespace-pre-wrap">
        {{ msg.content }}
      </div>
      <div v-else class="rounded-[8px] bg-surface-2 p-2">
        <textarea
          ref="editRef"
          v-model="draft"
          rows="2"
          class="input resize-none"
          @keydown.enter.exact.prevent="submitEdit"
        />
        <div class="mt-1.5 flex justify-end gap-1.5">
          <NButton quaternary size="tiny" @click="editing = false">{{ t('common.cancel') }}</NButton>
          <NButton type="primary" size="tiny" :disabled="busy" @click="submitEdit">{{ t('chat.regenerate') }}</NButton>
        </div>
      </div>
      <div class="mt-1 flex items-center justify-end gap-2 px-1 text-faint">
        <BranchNav
          v-if="branch.total > 1"
          :index="branch.index"
          :total="branch.total"
          @prev="prevBranch"
          @next="nextBranch"
        />
        <NButton
          v-if="!editing && !busy"
          text
          size="tiny"
          class="opacity-0 transition-opacity group-hover:opacity-100"
          :title="t('chat.editAndResend')"
          @click="startEdit"
        >
          <template #icon>
            <span class="i-lucide-pencil text-xs" />
          </template>
        </NButton>
      </div>
    </div>
  </div>

  <!-- 助手消息：直接渲染在背景上，不套盒子（低噪音） -->
  <div v-else class="flex justify-start">
    <div class="max-w-[88%] min-w-0">
      <div class="text-sm">
        <div v-if="msg.stage" class="flex items-center gap-2 text-muted">
          <span class="i-lucide-loader-circle animate-spin" /> {{ msg.stage }}
        </div>

        <!-- 思考过程（专家模式） -->
        <div v-if="msg.thinking" class="mb-2 rounded-[8px] bg-surface-2">
          <NButton
            quaternary
            block
            size="tiny"
            class="justify-start"
            @click="thinkingOpen = !thinkingOpen"
          >
            <template #icon>
              <span :class="isStreaming ? 'i-lucide-loader-circle animate-spin' : 'i-lucide-brain'" />
            </template>
            <span class="font-medium">{{ t('chat.thinking') }}</span>
            <span class="ml-auto">{{ thinkingOpen ? t('chat.collapse') : t('chat.expand') }}</span>
          </NButton>
          <div
            v-if="thinkingOpen"
            class="max-h-72 overflow-y-auto whitespace-pre-wrap border-t border-softer px-3 py-2 text-xs leading-relaxed text-muted"
          >
            {{ msg.thinking }}
          </div>
        </div>

        <div v-if="msg.error" class="text-sm text-red-500">⚠ {{ msg.error }}</div>

        <MarkdownMessage
          v-if="msg.content"
          :content="rendered"
          :streaming="isStreaming"
          @cite="onCite"
        />

        <ResearchBadges
          v-if="!isStreaming && msg.research && msg.research.rounds > 0"
          :research="msg.research"
        />
      </div>

      <!-- footer -->
      <div
        v-if="msg.status !== 'streaming'"
        class="mt-1 flex items-center gap-3 px-1 text-xs text-faint"
      >
        <NButton
          v-if="(msg.hits?.length ?? 0) > 0"
          text
          size="tiny"
          @click="emit('sources', msg.id)"
        >
          <template #icon>
            <span class="i-lucide-quote" />
          </template>
          {{ parsed.hasCitations ? t('chat.citationsCount', { n: sourceCount }) : t('chat.sourcesCount', { n: sourceCount }) }}
        </NButton>
        <span v-if="msg.latency != null">· {{ msg.latency.toFixed(2) }}s</span>
        <NButton
          v-if="!busy && msg.parentId"
          text
          size="tiny"
          :title="t('chat.regenerate')"
          @click="emit('regenerate', msg.id)"
        >
          <template #icon>
            <span class="i-lucide-refresh-cw" />
          </template>
          {{ t('chat.regenerate') }}
        </NButton>
        <BranchNav
          v-if="branch.total > 1"
          :index="branch.index"
          :total="branch.total"
          @prev="prevBranch"
          @next="nextBranch"
        />
      </div>
    </div>
  </div>
</template>
