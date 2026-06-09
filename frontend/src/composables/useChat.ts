import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from './useApi'
import { useConversationsStore } from '@/stores/conversations'
import type {
  ChatRequest,
  Conversation,
  RetrievalMode,
  RetrievalSourceKey,
  StreamEvent,
} from '@/api/types'

export interface ComposerState {
  collection: string
  docIds: string[]
  enableRetrieval: boolean
  professional: boolean
  useAgentic: boolean
  stream: boolean
  mode: RetrievalMode | 'auto'
  topK: number
  sources: RetrievalSourceKey[]
}

/**
 * 问答编排：发送 / 停止 / 重生成 / 编辑历史重生成（分叉）。
 * 当前走后端 /chat/stream（带 access_token）；后端 M4/M6 落地后切换到消息树端点与重连续读。
 */
export function useChat() {
  const api = useApi()
  const store = useConversationsStore()
  const { t } = useI18n()
  const busy = ref(false)
  let abort: AbortController | null = null

  function buildRequest(
    conv: Conversation,
    query: string,
    c: ComposerState,
    userMessageId: string,
    assistantMessageId: string,
  ): ChatRequest {
    const userMessage = conv.messages[userMessageId]
    return {
      query,
      session_id: conv.sessionId,
      mode: c.mode === 'auto' ? null : c.mode,
      top_k: c.topK,
      use_agentic: c.useAgentic,
      professional: c.professional,
      enable_retrieval: c.enableRetrieval,
      collection: c.collection || null,
      doc_ids: c.docIds.length > 0 ? c.docIds : null,
      sources: c.sources,
      conversation_id: conv.id,
      parent_message_id: userMessage?.parentId ?? null,
      client_user_message_id: userMessageId,
      client_assistant_message_id: assistantMessageId,
    }
  }

  function stageLabel(stage: string, expert: boolean): string {
    if (stage === 'generating') return expert ? t('chat.statusWriting') : t('chat.statusGenerating')
    return expert ? t('chat.statusResearching') : t('chat.statusRetrieving')
  }

  async function run(
    conv: Conversation,
    userMessageId: string,
    assistantId: string,
    query: string,
    c: ComposerState,
  ) {
    busy.value = true
    abort = new AbortController()
    store.patch(conv, assistantId, {
      content: '',
      thinking: '',
      error: undefined,
      status: 'streaming',
      stage: t('chat.statusRetrieving'),
    })
    const expert = c.professional

    const onEvent = (ev: StreamEvent) => {
      const m = conv.messages[assistantId]
      if (!m) return
      switch (ev.type) {
        case 'status':
          store.patch(conv, assistantId, { stage: stageLabel(ev.stage, expert) })
          break
        case 'thinking': {
          const prev = m.thinking ?? ''
          const sep = prev && ev.phase !== 'synthesis' && !prev.endsWith('\n') ? '\n\n' : ''
          store.patch(conv, assistantId, {
            thinking: prev + sep + ev.content,
            stage: m.content ? undefined : t('chat.thinking'),
          })
          break
        }
        case 'text':
          store.patch(conv, assistantId, { content: m.content + ev.content, stage: undefined })
          break
        case 'done':
          if (ev.session_id) conv.sessionId = ev.session_id
          store.patch(conv, assistantId, {
            content: ev.answer && ev.answer.length > 0 ? ev.answer : m.content,
            hits: ev.hits ?? [],
            context: typeof ev.context === 'string' ? ev.context : undefined,
            research: ev.research ?? null,
            latency: ev.latency_s,
            usage: ev.usage,
            status: 'done',
            stage: undefined,
          })
          break
        case 'error':
          store.patch(conv, assistantId, { status: 'failed', error: ev.message, stage: undefined })
          break
      }
    }

    try {
      await api.chatStream(buildRequest(conv, query, c, userMessageId, assistantId), onEvent, abort.signal)
      const m = conv.messages[assistantId]
      if (m && m.status === 'streaming') store.patch(conv, assistantId, { status: 'done', stage: undefined })
    } catch (e) {
      const m = conv.messages[assistantId]
      if (abort?.signal.aborted) {
        store.patch(conv, assistantId, { status: 'stopped', stage: undefined })
      } else if (m) {
        store.patch(conv, assistantId, {
          status: 'failed',
          stage: undefined,
          error: e instanceof Error ? e.message : String(e),
        })
      }
    } finally {
      busy.value = false
      abort = null
      store.persist()
    }
  }

  async function send(conv: Conversation, query: string, c: ComposerState) {
    if (busy.value || !query.trim()) return
    const { userId, assistantId } = store.appendTurn(conv, query.trim(), c.professional)
    await run(conv, userId, assistantId, query.trim(), c)
  }

  async function regenerate(conv: Conversation, assistantId: string, c: ComposerState) {
    if (busy.value) return
    const msg = conv.messages[assistantId]
    if (!msg || msg.parentId == null) return
    const userMsg = conv.messages[msg.parentId]
    if (!userMsg) return
    const newId = store.regenerate(conv, assistantId, c.professional)
    if (newId) await run(conv, userMsg.id, newId, userMsg.content, c)
  }

  async function editAndResend(conv: Conversation, userId: string, newContent: string, c: ComposerState) {
    if (busy.value || !newContent.trim()) return
    const forked = store.forkUser(conv, userId, newContent.trim(), c.professional)
    if (forked) await run(conv, forked.userId, forked.assistantId, newContent.trim(), c)
  }

  function stop() {
    abort?.abort()
  }

  return { busy, send, regenerate, editAndResend, stop }
}
