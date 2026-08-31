<!-- 类似问题推荐：点击作为新消息发送，可忽略 -->
<script lang="ts" setup>
import { ref } from "vue";

defineOptions({ name: "SuggestionList" });

defineProps<{
  questions: string[];
}>();

const emit = defineEmits<{
  apply: [question: string];
}>();

const dismissed = ref(false);
</script>

<template>
  <div v-if="!dismissed" class="suggestion-list">
    <span class="suggestion-list__label">相关问题：</span>
    <el-tag
      v-for="question in questions"
      :key="question"
      class="suggestion-list__item"
      effect="plain"
      @click="emit('apply', question)"
    >
      {{ question }}
    </el-tag>
    <el-button link size="small" @click="dismissed = true">忽略</el-button>
  </div>
</template>

<style scoped lang="scss">
.suggestion-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  max-width: 92%;
  margin-top: 8px;

  &__label {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__item {
    cursor: pointer;
  }
}
</style>
