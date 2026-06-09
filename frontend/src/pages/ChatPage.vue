<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import { useConversationsStore } from '@/stores/conversations'
import { useChat, type ComposerState } from '@/composables/useChat'
import { useSettings } from '@/composables/useSettings'
import { parseCitations } from '@/utils/citations'
import type { Conversation } from '@/api/types'
import type { SourceItem } from '@/components/SourcesPanel.vue'
import WorkspaceSplit from '@/components/WorkspaceSplit.vue'
import ConversationList from '@/components/ConversationList.vue'
import MessageBubble from '@/components/MessageBubble.vue'
import ChatComposer from '@/components/ChatComposer.vue'
import SourcesPanel from '@/components/SourcesPanel.vue'

const { t } = useI18n()
const store = useConversationsStore()
const api = useApi()
const settings = useSettings()
const chat = useChat()

const composer = reactive<ComposerState>({
  collection: '',
  docIds: [],
  enableRetrieval: settings.value.enableRetrieval,
  professional: settings.value.professional,
  useAgentic: settings.value.useAgentic,
  stream: settings.value.stream,
  mode: settings.value.mode,
  topK: settings.value.topK,
  sources: [...settings.value.sources],
})

const threadRef = ref<HTMLElement | null>(null)
const sourcesForId = ref<string | null>(null)
const panelTab = ref<'summary' | 'chunks' | 'pdf'>('summary')
const highlightNum = ref<number | null>(null)
const sharing = ref(false)

onMounted(() => {
  store.load()
  if (!store.activeId) {
    const first = store.list()[0]
    store.activeId = first ? first.id : null
  }
})

const activeConv = computed<Conversation | null>(() =>
  store.activeId ? store.get(store.activeId) ?? null : null,
)
const messages = computed(() => (activeConv.value ? store.mainline(activeConv.value) : []))

function ensureConv(): Conversation {
  if (activeConv.value) return activeConv.value
  return store.create()
}

async function onSend(query: string) {
  const conv = ensureConv()
  await chat.send(conv, query, composer)
}
function onStop() {
  chat.stop()
}
async function onRegenerate(assistantId: string) {
  if (activeConv.value) await chat.regenerate(activeConv.value, assistantId, composer)
}
async function onEdit(userId: string, content: string) {
  if (activeConv.value) await chat.editAndResend(activeConv.value, userId, content, composer)
}

function openSources(msgId: string) {
  sourcesForId.value = msgId
  panelTab.value = 'chunks'
  highlightNum.value = null
}
function onCite(msgId: string, num: number) {
  sourcesForId.value = msgId
  panelTab.value = 'pdf'
  highlightNum.value = num
}

async function onShare() {
  const conv = activeConv.value
  if (!conv || sharing.value) return
  sharing.value = true
  try {
    if (conv.shareToken) {
      await api.unshareConversation(conv.id)
      conv.shareToken = null
      store.persist()
      return
    }
    const share = await api.shareConversation(conv.id)
    conv.shareToken = share.token
    store.persist()
    await navigator.clipboard?.writeText(share.url)
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  } finally {
    sharing.value = false
  }
}

const sourceItems = computed<SourceItem[]>(() => {
  const conv = activeConv.value
  if (!conv || !sourcesForId.value) return []
  const m = conv.messages[sourcesForId.value]
  if (!m) return []
  const hits = m.hits ?? []
  const parsed = parseCitations(m.content, hits, m.context, { research: m.expert })
  if (parsed.hasCitations) return parsed.citedHits.map((c) => ({ num: c.num, hit: c.hit }))
  return hits.map((h, i) => ({ num: i + 1, hit: h }))
})

// 切换会话或新消息时滚到底部，关闭来源面板
watch(() => store.activeId, () => {
  sourcesForId.value = null
})
watch(
  () => messages.value.map((m) => m.content).join('|'),
  async () => {
    await nextTick()
    threadRef.value?.scrollTo({ top: threadRef.value.scrollHeight, behavior: 'smooth' })
  },
)
</script>

<template>
  <div class="flex h-full min-w-0">
    <WorkspaceSplit>
      <template #pane1>
        <div class="workspace-card">
          <ConversationList :busy="chat.busy.value" />
        </div>
      </template>

      <template #pane2>
        <div class="workspace-card flex flex-col">
          <header class="flex items-center justify-between border-b border-softer px-6 py-3.5">
            <div class="flex items-center gap-2 text-sm font-semibold text-base">
              {{ t('chat.title') }}
              <span v-if="activeConv?.sessionId" class="chip font-normal">
                {{ activeConv.sessionId.slice(0, 8) }}
              </span>
            </div>
            <NButton
              v-if="activeConv && messages.length > 0"
              quaternary
              size="small"
              :loading="sharing"
              @click="onShare"
            >
              <template #icon>
                <span :class="activeConv.shareToken ? 'i-lucide-unlink' : 'i-lucide-share-2'" />
              </template>
              {{ activeConv.shareToken ? t('chat.unshare') : t('chat.share') }}
            </NButton>
          </header>

          <div ref="threadRef" class="min-h-0 flex-1 overflow-y-auto px-6 py-7">
            <div v-if="messages.length === 0" class="grid h-full place-items-center text-center">
              <div>
                <div class="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-[8px] bg-surface-2 text-accent">
                  <span class="i-lucide-sparkles text-xl" />
                </div>
                <h2 class="text-base font-semibold">{{ t('chat.emptyTitle') }}</h2>
                <p class="mt-1 max-w-md text-sm text-muted">{{ t('chat.emptySubtitle') }}</p>
              </div>
            </div>
            <div v-else class="mx-auto flex max-w-3xl flex-col gap-6">
              <MessageBubble
                v-for="m in messages"
                :key="m.id"
                :msg="m"
                :conv="activeConv!"
                :busy="chat.busy.value"
                @sources="openSources"
                @cite="onCite"
                @regenerate="onRegenerate"
                @edit="onEdit"
              />
            </div>
          </div>

          <ChatComposer :composer="composer" :busy="chat.busy.value" @send="onSend" @stop="onStop" />
        </div>
      </template>
    </WorkspaceSplit>

    <SourcesPanel
      v-if="sourcesForId"
      :items="sourceItems"
      :tab="panelTab"
      :highlight-num="highlightNum"
      @close="sourcesForId = null"
      @update:tab="panelTab = $event"
    />
  </div>
</template>
