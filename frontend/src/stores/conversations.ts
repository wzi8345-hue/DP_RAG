import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ChatMessage, Conversation, Visibility } from '@/api/types'

// 前端消息树（localStorage 持久化）。后端 M4 落地后切换为 Postgres conversations/messages。
// 主线推导：从 activeLeafId 沿 parentId 递归到根，再反转。
// 分叉：编辑历史 user 消息 → 同 parentId 新建兄弟分支；重生成 assistant → 同 parentId 新建兄弟。

const STORAGE_KEY = 'dp-rag-conversations'
const MAX_CONVERSATIONS = 50

function rid(prefix = 'm'): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

export const useConversationsStore = defineStore('conversations', () => {
  const conversations = ref<Record<string, Conversation>>({})
  const activeId = ref<string | null>(null)

  function load(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) conversations.value = JSON.parse(raw)
    } catch {
      conversations.value = {}
    }
  }

  function persist(): void {
    try {
      const entries = Object.values(conversations.value)
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_CONVERSATIONS)
      const trimmed: Record<string, Conversation> = {}
      for (const c of entries) trimmed[c.id] = c
      conversations.value = trimmed
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
    } catch {
      /* quota */
    }
  }

  function list(): Conversation[] {
    return Object.values(conversations.value).sort((a, b) => b.updatedAt - a.updatedAt)
  }

  function get(id: string): Conversation | undefined {
    return conversations.value[id]
  }

  function create(): Conversation {
    const conv: Conversation = {
      id: rid('c'),
      title: '',
      sessionId: null,
      visibility: 'private',
      messages: {},
      rootIds: [],
      activeLeafId: null,
      updatedAt: Date.now(),
    }
    conversations.value[conv.id] = conv
    activeId.value = conv.id
    return conv
  }

  function remove(id: string): void {
    delete conversations.value[id]
    if (activeId.value === id) activeId.value = null
    persist()
  }

  function touch(conv: Conversation): void {
    conv.updatedAt = Date.now()
    if (!conv.title) {
      const firstUser = mainline(conv).find((m) => m.role === 'user')
      if (firstUser) {
        const t = firstUser.content.trim().replace(/\s+/g, ' ')
        conv.title = t.length > 28 ? `${t.slice(0, 28)}…` : t || '新对话'
      }
    }
  }

  function childrenOf(conv: Conversation, parentId: string | null): ChatMessage[] {
    const arr = parentId === null
      ? conv.rootIds.map((id) => conv.messages[id])
      : Object.values(conv.messages).filter((m) => m.parentId === parentId)
    return arr.filter(Boolean).sort((a, b) => a.createdAt - b.createdAt)
  }

  function deepestLeaf(conv: Conversation, fromId: string): string {
    let cur = fromId
    for (;;) {
      const kids = childrenOf(conv, cur)
      if (kids.length === 0) return cur
      cur = kids[kids.length - 1].id
    }
  }

  function mainline(conv: Conversation): ChatMessage[] {
    const chain: ChatMessage[] = []
    let mid = conv.activeLeafId
    const seen = new Set<string>()
    while (mid && conv.messages[mid] && !seen.has(mid)) {
      seen.add(mid)
      const m = conv.messages[mid]
      chain.push(m)
      mid = m.parentId
    }
    return chain.reverse()
  }

  function addMessage(conv: Conversation, msg: ChatMessage): void {
    conv.messages[msg.id] = msg
    if (msg.parentId === null) {
      if (!conv.rootIds.includes(msg.id)) conv.rootIds.push(msg.id)
    }
    conv.activeLeafId = msg.id
  }

  /** 追加一轮：user 消息（parent=当前叶子）+ assistant 占位，返回二者 id。 */
  function appendTurn(
    conv: Conversation,
    userContent: string,
    expert: boolean,
  ): { userId: string; assistantId: string } {
    const parentId = conv.activeLeafId
    const user: ChatMessage = {
      id: rid(),
      parentId,
      role: 'user',
      content: userContent,
      createdAt: Date.now(),
    }
    addMessage(conv, user)
    const assistant: ChatMessage = {
      id: rid(),
      parentId: user.id,
      role: 'assistant',
      content: '',
      status: 'streaming',
      expert,
      createdAt: Date.now() + 1,
    }
    addMessage(conv, assistant)
    touch(conv)
    return { userId: user.id, assistantId: assistant.id }
  }

  /** 编辑历史 user 消息并重生成：在该 user 的 parentId 下新建兄弟 user + assistant 占位（分叉）。 */
  function forkUser(
    conv: Conversation,
    userMessageId: string,
    newContent: string,
    expert: boolean,
  ): { userId: string; assistantId: string } | null {
    const old = conv.messages[userMessageId]
    if (!old || old.role !== 'user') return null
    const user: ChatMessage = {
      id: rid(),
      parentId: old.parentId,
      role: 'user',
      content: newContent,
      createdAt: Date.now(),
    }
    addMessage(conv, user)
    const assistant: ChatMessage = {
      id: rid(),
      parentId: user.id,
      role: 'assistant',
      content: '',
      status: 'streaming',
      expert,
      createdAt: Date.now() + 1,
    }
    addMessage(conv, assistant)
    touch(conv)
    return { userId: user.id, assistantId: assistant.id }
  }

  /** 重生成 assistant：在其 parent(user) 下新建兄弟 assistant 占位（分叉）。 */
  function regenerate(conv: Conversation, assistantId: string, expert: boolean): string | null {
    const old = conv.messages[assistantId]
    if (!old || old.role !== 'assistant' || old.parentId == null) return null
    const assistant: ChatMessage = {
      id: rid(),
      parentId: old.parentId,
      role: 'assistant',
      content: '',
      status: 'streaming',
      expert,
      createdAt: Date.now(),
    }
    addMessage(conv, assistant)
    touch(conv)
    return assistant.id
  }

  function patch(conv: Conversation, msgId: string, p: Partial<ChatMessage>): void {
    const m = conv.messages[msgId]
    if (!m) return
    Object.assign(m, p)
    touch(conv)
  }

  /** 分支信息：某消息在其兄弟中的位置（用于上一/下一分支）。 */
  function branchInfo(conv: Conversation, msgId: string): { index: number; total: number; siblings: ChatMessage[] } {
    const m = conv.messages[msgId]
    if (!m) return { index: 0, total: 1, siblings: [] }
    const siblings = childrenOf(conv, m.parentId)
    const index = siblings.findIndex((s) => s.id === msgId)
    return { index, total: siblings.length, siblings }
  }

  /** 切换到某兄弟分支：activeLeaf 设为该兄弟最深叶子。 */
  function switchBranch(conv: Conversation, siblingId: string): void {
    conv.activeLeafId = deepestLeaf(conv, siblingId)
    touch(conv)
  }

  function setVisibility(conv: Conversation, v: Visibility): void {
    conv.visibility = v
    touch(conv)
    persist()
  }

  return {
    conversations,
    activeId,
    load,
    persist,
    list,
    get,
    create,
    remove,
    mainline,
    childrenOf,
    appendTurn,
    forkUser,
    regenerate,
    patch,
    branchInfo,
    switchBranch,
    setVisibility,
  }
})
