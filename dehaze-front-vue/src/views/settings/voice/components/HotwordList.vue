<template>
  <div class="hotword-list">
    <el-empty
      v-if="words.length === 0"
      description="暂无热词"
      :image-size="60"
    />
    <div v-for="hotword in words" :key="hotword.id" class="hotword-item">
      <span class="word-text">{{ hotword.word }}</span>
      <el-button link type="danger" @click="emit('delete', hotword.id)">
        <el-icon><Delete /></el-icon>
      </el-button>
    </div>
  </div>
</template>

<script lang="ts" setup>
import type { HotwordVO } from "dehaze-sdk-js";
import { Delete } from "@element-plus/icons-vue";

defineOptions({ name: "HotwordList" });

defineProps<{
  words: HotwordVO[];
}>();

const emit = defineEmits<{
  (e: "delete", id: number): void;
}>();
</script>

<style lang="scss" scoped>
.hotword-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);

  &:last-child {
    border-bottom: none;
  }

  .word-text {
    font-size: 14px;
    color: var(--el-text-color-primary);
  }
}
</style>
