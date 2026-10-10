<!-- 知识库卡片 -->
<script lang="ts" setup>
import type { KnowledgeBaseVO } from "dehaze-sdk-js";
import { computed } from "vue";
import { formatTokens } from "@/utils/format";

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

/** 卡片「···」菜单分发（破坏性操作集中在此） */
function handleCommand(command: string) {
  if (command === "delete") {
    emit("delete", props.kb);
  }
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
      <div v-if="canManage" class="kb-card__actions" @click.stop>
        <el-button type="primary" link size="small" @click="emit('edit', kb)">
          编辑
        </el-button>
        <!-- 删除是破坏性操作：收进「···」并二次确认，避免红字常驻误触（设计评审 #14） -->
        <el-dropdown trigger="click" @command="handleCommand">
          <el-button link size="small" class="kb-card__more">···</el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="delete">删除</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </div>
    <p class="kb-desc">
      {{ kb.description || "暂无描述，点击卡片进入可补充" }}
    </p>
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
.kb-card__actions {
  display: flex;
  gap: 4px;
  align-items: center;
}

.kb-card__more {
  letter-spacing: 1px;
}

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
