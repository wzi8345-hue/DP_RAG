<script setup lang="ts">
import MarkdownRender from 'markstream-vue'

defineProps<{ content: string; streaming?: boolean }>()
const emit = defineEmits<{ (e: 'cite', num: number): void }>()

function onClick(e: MouseEvent) {
  const a = (e.target as HTMLElement).closest('a')
  const href = a?.getAttribute('href')
  if (href && href.startsWith('#cite-')) {
    e.preventDefault()
    emit('cite', Number(href.slice('#cite-'.length)))
  }
}
</script>

<template>
  <div class="markstream-host" @click="onClick">
    <MarkdownRender
      :content="content"
      :enable-katex="true"
      :max-live-nodes="0"
      :typewriter="!!streaming"
    />
  </div>
</template>
