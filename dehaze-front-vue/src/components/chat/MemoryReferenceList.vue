<!-- 助手回复引用的记忆清单（默认折叠） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { MemoryVO } from "dehaze-sdk-js";

defineOptions({ name: "MemoryReferenceList" });

const props = defineProps<{
  memories: MemoryVO[];
}>();

const typeLabels: Record<string, string> = {
  episodic: "情景",
  semantic: "语义",
  procedural: "程序",
};

const items = computed(() =>
  props.memories.map((memory) => ({
    id: memory.id,
    typeLabel: typeLabels[memory.memoryType] ?? memory.memoryType,
    content: memory.content,
    importance: memory.importance,
  }))
);
</script>

<template>
  <el-collapse class="memory-reference-list">
    <el-collapse-item
      :title="`引用记忆（${memories.length} 条）`"
      name="memories"
    >
      <div
        v-for="item in items"
        :key="item.id"
        class="memory-reference-list__item"
      >
        <el-tag size="small" type="info">{{ item.typeLabel }}</el-tag>
        <span class="memory-reference-list__content">{{ item.content }}</span>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped lang="scss">
.memory-reference-list {
  max-width: 92%;
  margin-bottom: 8px;

  :deep(.el-collapse-item__header) {
    height: 32px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 4px 0;
  }

  &__content {
    font-size: 13px;
    color: var(--el-text-color-regular);
    overflow-wrap: anywhere;
  }
}
</style>
