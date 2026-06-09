<script setup lang="ts">
import { NSplit } from 'naive-ui'

withDefaults(defineProps<{
  defaultSize?: string
  min?: string
  max?: string
  resizeTriggerSize?: number
  pane1Class?: string
  pane2Class?: string
  triggerTitle?: string
}>(), {
  defaultSize: '264px',
  min: '220px',
  max: '360px',
  resizeTriggerSize: 8,
  pane1Class: '',
  pane2Class: '',
  triggerTitle: '拖拽调整宽度',
})
</script>

<template>
  <NSplit
    class="workspace-split"
    direction="horizontal"
    :default-size="defaultSize"
    :min="min"
    :max="max"
    :resize-trigger-size="resizeTriggerSize"
    :pane1-class="`min-w-0 ${pane1Class}`"
    :pane2-class="`min-w-0 ${pane2Class}`"
  >
    <template #1>
      <slot name="pane1" />
    </template>

    <template #resize-trigger>
      <div class="workspace-split-trigger" :title="triggerTitle" />
    </template>

    <template #2>
      <slot name="pane2" />
    </template>
  </NSplit>
</template>
