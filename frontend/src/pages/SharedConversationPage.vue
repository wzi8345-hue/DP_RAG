<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useLogto } from '@logto/vue'
import { NButton } from 'naive-ui'
import { useApi } from '@/composables/useApi'
import { redirectUri } from '@/auth/logto'
import { useConversationsStore } from '@/stores/conversations'
import MarkdownMessage from '@/components/MarkdownMessage.vue'

const route = useRoute()
const router = useRouter()
const api = useApi()
const store = useConversationsStore()
const { isAuthenticated, signIn } = useLogto()

const loading = ref(true)
const copying = ref(false)
const error = ref('')
const conversation = ref<Record<string, unknown> | null>(null)

const token = computed(() => String(route.params.token || ''))
const messages = computed(() => {
  const conv = conversation.value
  const raw = conv?.messages
  if (!raw || typeof raw !== 'object') return []
  const byId = raw as Record<string, Record<string, unknown>>
  const activeLeaf = typeof conv?.activeLeafId === 'string' ? conv.activeLeafId : null
  const chain: Record<string, unknown>[] = []
  let cur = activeLeaf
  const seen = new Set<string>()
  while (cur && byId[cur] && !seen.has(cur)) {
    seen.add(cur)
    chain.push(byId[cur])
    const parentId = byId[cur].parentId
    cur = typeof parentId === 'string' ? parentId : null
  }
  return chain.reverse()
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    const res = await api.getSharedConversation(token.value)
    conversation.value = res.conversation as Record<string, unknown>
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function continueConversation() {
  if (!isAuthenticated.value) {
    await signIn(redirectUri)
    return
  }
  copying.value = true
  try {
    const copied = await api.copySharedConversationToMine(token.value)
    const res = await api.getConversation(copied.conversation_id)
    store.importBackend(res.conversation as Record<string, unknown>)
    await router.push('/chat')
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    copying.value = false
  }
}

onMounted(load)
</script>

<template>
  <main class="h-full overflow-y-auto bg-app px-4 py-8">
    <section class="mx-auto flex max-w-3xl flex-col gap-5 rounded-[10px] bg-surface p-6">
      <header class="flex items-start justify-between gap-4 border-b border-softer pb-4">
        <div class="min-w-0">
          <div class="text-xs text-faint">Shared conversation</div>
          <h1 class="truncate text-lg font-semibold text-base">
            {{ conversation?.title || 'Conversation' }}
          </h1>
        </div>
        <NButton type="primary" :loading="copying" @click="continueConversation">
          <template #icon>
            <span class="i-lucide-message-square-plus" />
          </template>
          Continue
        </NButton>
      </header>

      <div v-if="loading" class="py-12 text-center text-sm text-faint">
        Loading…
      </div>
      <div v-else-if="error" class="rounded-[8px] bg-surface-2 p-4 text-sm text-red-500">
        {{ error }}
      </div>
      <div v-else class="flex flex-col gap-5">
        <article
          v-for="m in messages"
          :key="String(m.id)"
          class="rounded-[8px] p-3"
          :class="m.role === 'user' ? 'ml-auto max-w-[80%] bg-accent text-white' : 'bg-transparent text-base'"
        >
          <div v-if="m.role === 'user'" class="whitespace-pre-wrap text-sm">
            {{ m.content }}
          </div>
          <MarkdownMessage v-else :content="String(m.content || '')" />
        </article>
      </div>
    </section>
  </main>
</template>
