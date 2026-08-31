<!-- 私有库监控表：只读审计视角，无详情跳转与内容操作入口 -->
<script lang="ts" setup>
import {
  AiKnowledgeBaseAPI,
  IndexStatsVO,
  KnowledgeBaseVO,
} from "dehaze-sdk-js";
import { reactive } from "vue";

defineOptions({ name: "PrivateKbMonitorTable" });

defineProps<{
  kbs: KnowledgeBaseVO[];
  loading: boolean;
}>();

// 索引状态按行懒加载：展开"查看索引状态"时再拉取，避免列表进入即批量请求
const statsMap = reactive<Record<number, IndexStatsVO | null>>({});
const statsLoading = reactive<Record<number, boolean>>({});

async function loadStats(kbId: number) {
  if (statsLoading[kbId]) return;
  statsLoading[kbId] = true;
  try {
    statsMap[kbId] = await AiKnowledgeBaseAPI.getIndexStats(kbId);
  } finally {
    statsLoading[kbId] = false;
  }
}

function formatSize(bytes: number) {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
</script>

<template>
  <el-table v-loading="loading" :data="kbs" border row-key="id">
    <el-table-column
      label="知识库名称"
      prop="name"
      min-width="180"
      show-overflow-tooltip
    />
    <el-table-column
      label="创建者 ID"
      prop="createBy"
      width="110"
      align="center"
    >
      <template #default="{ row }">
        <span v-if="(row as KnowledgeBaseVO).createBy">{{
          (row as KnowledgeBaseVO).createBy
        }}</span>
        <span v-else class="text-muted">-</span>
      </template>
    </el-table-column>
    <el-table-column
      label="文档数"
      prop="documentCount"
      width="90"
      align="center"
    />
    <el-table-column
      label="分块数"
      prop="chunkCount"
      width="90"
      align="center"
    />
    <el-table-column
      label="Embedding 模型"
      prop="embeddingModel"
      width="160"
      show-overflow-tooltip
    />
    <el-table-column
      label="更新时间"
      prop="updateTime"
      width="170"
      align="center"
    >
      <template #default="{ row }">
        <span>{{ (row as KnowledgeBaseVO).updateTime || "-" }}</span>
      </template>
    </el-table-column>
    <el-table-column type="expand" width="120" label="索引状态">
      <template #default="{ row }">
        <div class="stats-expand">
          <template v-if="statsMap[(row as KnowledgeBaseVO).id]">
            <el-tag
              :type="
                statsMap[(row as KnowledgeBaseVO).id]!.thresholdWarning
                  ? 'danger'
                  : 'success'
              "
              size="small"
            >
              {{
                statsMap[(row as KnowledgeBaseVO).id]!.thresholdWarning
                  ? "阈值告警"
                  : "健康"
              }}
            </el-tag>
            <span
              >索引大小：{{
                formatSize(statsMap[(row as KnowledgeBaseVO).id]!.indexSize)
              }}</span
            >
            <span
              >索引文档数：{{
                statsMap[(row as KnowledgeBaseVO).id]!.indexDocCount
              }}</span
            >
          </template>
          <el-button
            v-else
            size="small"
            link
            type="primary"
            :loading="statsLoading[(row as KnowledgeBaseVO).id]"
            @click="loadStats((row as KnowledgeBaseVO).id)"
          >
            查看索引状态
          </el-button>
        </div>
      </template>
    </el-table-column>
  </el-table>
</template>

<style scoped>
.stats-expand {
  display: flex;
  gap: 16px;
  align-items: center;
  padding: 8px 16px;
}

.text-muted {
  color: var(--el-text-color-placeholder);
}
</style>
