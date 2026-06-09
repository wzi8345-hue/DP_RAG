<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { NButton, NInput, NVirtualList } from 'naive-ui'
import { useConversationsStore } from '@/stores/conversations'
import type { Conversation } from '@/api/types'

const props = defineProps<{ busy: boolean }>()
const { t } = useI18n()
const store = useConversationsStore()
const searchDraft = ref('')
const keyword = ref('')

const items = computed(() => {
  const q = keyword.value.trim().toLowerCase()
  const all = store.list()
  if (!q) return all
  return all.filter((c) => (c.title || t('nav.newChat')).toLowerCase().includes(q))
})

function newChat() {
  if (props.busy) return
  store.create()
}
function search() {
  keyword.value = searchDraft.value
}
function select(id: string) {
  if (props.busy) return
  store.activeId = id
}
function remove(id: string, e: Event) {
  e.stopPropagation()
  store.remove(id)
}
</script>

<template>
  <div class="flex h-full min-w-0 flex-col">
    <div class="flex flex-col gap-2 p-2">
      <NButton tertiary size="medium" block class="justify-start" :disabled="busy" :title="t('nav.newChat')" @click="newChat">
        <template #icon>
          <span class="i-lucide-plus" />
        </template>
        {{ t('nav.newChat') }}
      </NButton>
      <NInput
        v-model:value="searchDraft"
        
        clearable
        placeholder="搜索对话"
        @keydown.enter="search"
        @clear="keyword = ''"
      >
        <template #suffix>
          <NButton text size="tiny" title="搜索对话" @click="search">
            <template #icon>
              <span class="i-lucide-search" />
            </template>
          </NButton>
        </template>
      </NInput>
    </div>

    <div class="menu-section-divider" />

    <div class="min-h-0 flex-1">
      <p v-if="items.length === 0" class="px-2 py-4 text-center text-xs text-faint">
        {{ t('nav.noConversations') }}
      </p>
      <NVirtualList
        v-else
        :items="items"
        :item-size="40"
        key-field="id"
        class="h-full"
        :padding-top="8"
        :padding-bottom="8"
      >
        <template #default="{ item }: { item: Conversation }">
          <div
            class="group mx-2 mb-0.5 flex h-9 cursor-pointer items-center gap-1 rounded-[6px] px-2 text-sm transition-colors"
            :class="store.activeId === item.id ? 'bg-active text-base hover:bg-active-hover' : 'text-muted hover:bg-hover hover:text-base'"
            @click="select(item.id)"
          >
            <span :class="item.visibility === 'org' ? 'i-lucide-users' : 'i-lucide-message-square'" class="shrink-0 text-xs" />
            <span class="min-w-0 flex-1 truncate" :title="item.title">{{ item.title || t('nav.newChat') }}</span>
            <NButton
              quaternary
              circle
              size="tiny"
              class="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
              :title="t('common.delete')"
              @click="remove(item.id, $event)"
            >
              <template #icon>
                <span class="i-lucide-trash-2 text-xs text-red-500" />
              </template>
            </NButton>
          </div>
        </template>
      </NVirtualList>
    </div>
  </div>
</template>
