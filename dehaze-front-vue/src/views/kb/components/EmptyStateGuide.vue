<!-- 空状态引导：新用户串联创建闭环 -->
<script lang="ts" setup>
defineOptions({ name: "EmptyStateGuide" });

defineProps<{
  isPublic?: boolean;
}>();

const emit = defineEmits<{
  (e: "create"): void;
}>();
</script>

<template>
  <el-empty
    :description="isPublic ? '暂无公共知识库' : '还没有知识库'"
    :image-size="100"
  >
    <el-button v-if="!isPublic" type="primary" @click="emit('create')">
      创建我的第一个知识库
    </el-button>
  </el-empty>
</template>
