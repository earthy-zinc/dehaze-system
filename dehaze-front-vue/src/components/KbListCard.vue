<!-- 知识库卡片 -->
<script lang="ts" setup>
import type { KnowledgeBaseVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "KbListCard" });

const props = defineProps<{
  kb: KnowledgeBaseVO;
  scope: "self" | "admin";
}>();

const emit = defineEmits<{
  (e: "click", kb: KnowledgeBaseVO): void;
  (e: "edit", kb: KnowledgeBaseVO): void;
  (e: "delete", kb: KnowledgeBaseVO): void;
}>();

// scope=self 公共库全员只读；私有库与管理端全量可管理
const canManage = computed(() => {
  if (props.scope === "admin") return true;
  return props.kb.visibility === "private";
});

function formatTokens(count: number): string {
  if (count >= 10000) {
    return `${(count / 10000).toFixed(1)}万`;
  }
  return String(count);
}
</script>

<template>
  <el-card
    shadow="hover"
    class="kb-card !border-none cursor-pointer"
    @click="emit('click', kb)"
  >
    <div class="flex items-start justify-between">
      <div class="flex flex-wrap items-center gap-2">
        <span class="text-base font-bold">{{ kb.name }}</span>
        <el-tag
          v-if="kb.visibility === 'public' && scope === 'self'"
          type="success"
          size="small"
        >
          平台公共知识库
        </el-tag>
        <el-tag v-else-if="kb.visibility === 'public'" type="info" size="small">
          公共
        </el-tag>
      </div>
      <div v-if="canManage" @click.stop>
        <el-button type="primary" link size="small" @click="emit('edit', kb)">
          编辑
        </el-button>
        <el-button type="danger" link size="small" @click="emit('delete', kb)">
          删除
        </el-button>
      </div>
    </div>
    <p class="kb-desc">{{ kb.description || "暂无描述" }}</p>
    <div class="kb-stats">
      <span>文档 {{ kb.documentCount }}</span>
      <el-divider direction="vertical" />
      <span>分块 {{ kb.chunkCount }}</span>
      <el-divider direction="vertical" />
      <span>Token {{ formatTokens(kb.totalTokens) }}</span>
    </div>
  </el-card>
</template>

<style lang="scss" scoped>
.kb-desc {
  display: -webkit-box;
  min-height: 36px;
  margin: 8px 0 12px;
  overflow: hidden;
  -webkit-line-clamp: 2;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  -webkit-box-orient: vertical;
}

.kb-stats {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
