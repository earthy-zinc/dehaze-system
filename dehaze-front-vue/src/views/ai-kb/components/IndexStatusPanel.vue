<!-- 索引状态区：索引大小/索引文档数/阈值告警 + 手动刷新 -->
<script lang="ts" setup>
import { storeToRefs } from "pinia";
import { ref } from "vue";
import { Refresh } from "@element-plus/icons-vue";
import { useAdminKbStore } from "@/store/modules/adminKb";

defineOptions({ name: "IndexStatusPanel" });

const props = defineProps<{
  kbId: number;
}>();

const adminKbStore = useAdminKbStore();
const { indexStats } = storeToRefs(adminKbStore);
const refreshing = ref(false);

async function refresh() {
  refreshing.value = true;
  try {
    await adminKbStore.fetchIndexStats(props.kbId);
  } finally {
    refreshing.value = false;
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
  <el-card shadow="never" class="!border-none">
    <template #header>
      <div class="flex justify-between items-center">
        <span>索引状态</span>
        <el-button link type="primary" :loading="refreshing" @click="refresh">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
    </template>
    <template v-if="indexStats">
      <el-alert
        v-if="indexStats.thresholdWarning"
        type="error"
        :closable="false"
        show-icon
        class="mb-4"
        title="索引大小已超过阈值（默认 1GB）"
        description="建议调整分块策略（减小分块大小/重叠）或清理低价值文档，控制索引规模。"
      />
      <el-descriptions :column="3" border>
        <el-descriptions-item label="索引大小">
          {{ formatSize(indexStats.indexSize) }}
        </el-descriptions-item>
        <el-descriptions-item label="索引文档数">
          {{ indexStats.indexDocCount }}
        </el-descriptions-item>
        <el-descriptions-item label="阈值告警">
          <el-tag
            :type="indexStats.thresholdWarning ? 'danger' : 'success'"
            size="small"
          >
            {{ indexStats.thresholdWarning ? "已触发" : "正常" }}
          </el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </template>
    <el-empty v-else description="索引状态未加载" :image-size="60" />
  </el-card>
</template>
