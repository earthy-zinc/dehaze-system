<!-- 空状态引导：新用户串联创建闭环；搜索无结果时给「清空搜索」而非误导性的创建引导 -->
<script lang="ts" setup>
defineOptions({ name: "EmptyStateGuide" });

defineProps<{
  isPublic?: boolean;
  /** 当前是否处于搜索/筛选态：搜索无结果与「一个都没有」是两回事，引导动作也不同 */
  searching?: boolean;
}>();

const emit = defineEmits<{
  (e: "create"): void;
  (e: "clear-search"): void;
}>();
</script>

<template>
  <el-empty
    :description="
      searching
        ? '没有匹配的知识库'
        : isPublic
          ? '暂无公共知识库'
          : '还没有知识库'
    "
    :image-size="100"
  >
    <template v-if="searching">
      <el-button @click="emit('clear-search')">清空搜索条件</el-button>
    </template>
    <template v-else-if="!isPublic">
      <el-button type="primary" @click="emit('create')">
        创建第一个知识库开始问答
      </el-button>
      <p class="empty-state-guide__hint">
        创建后可上传文档、做检索测试，并在 AI 对话中引用检索结果
      </p>
    </template>
    <template v-else>
      <p class="empty-state-guide__hint">
        平台暂未开放公共知识库，可先在「我的知识库」创建自己的库
      </p>
    </template>
  </el-empty>
</template>

<style scoped lang="scss">
.empty-state-guide__hint {
  margin-top: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
