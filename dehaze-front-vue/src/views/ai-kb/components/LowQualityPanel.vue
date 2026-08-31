<!-- 低质量片段：被点踩片段列表 + 处置（清理/重新分块） -->
<script lang="ts" setup>
import { LowQualityChunkVO } from "dehaze-sdk-js";
import { onMounted } from "vue";
import { useAdminKbStore } from "@/store/modules/adminKb";

defineOptions({ name: "LowQualityPanel" });

const props = defineProps<{
  kbId: number;
}>();

const adminKbStore = useAdminKbStore();
const disposingId = ref<number | null>(null);

async function handleDispose(
  chunk: LowQualityChunkVO,
  mode: "clean" | "rechunk"
) {
  const actionText = mode === "clean" ? "清理" : "重新分块";
  const tip =
    mode === "clean"
      ? `确认清理该低质量片段所在的文档「${chunk.documentTitle ?? `#${chunk.documentId}`}」？将删除文档及全部分块。`
      : `确认对「${chunk.documentTitle ?? `#${chunk.documentId}`}」触发重新分块？将重跑解析流水线并重建索引。`;
  try {
    await ElMessageBox.confirm(tip, `${actionText}确认`, {
      type: "warning",
      confirmButtonText: "确认",
      cancelButtonText: "取消",
    });
  } catch {
    return;
  }
  disposingId.value = chunk.chunkId;
  try {
    await adminKbStore.disposeLowQuality(props.kbId, chunk, mode);
    ElMessage.success(`${actionText}操作已提交`);
  } finally {
    disposingId.value = null;
  }
}

function handlePageChange(pageNum: number) {
  adminKbStore.lowQualityQuery.pageNum = pageNum;
  adminKbStore.fetchLowQuality(props.kbId);
}

function handleSizeChange(pageSize: number) {
  adminKbStore.lowQualityQuery.pageSize = pageSize;
  adminKbStore.lowQualityQuery.pageNum = 1;
  adminKbStore.fetchLowQuality(props.kbId);
}

onMounted(() => {
  adminKbStore.fetchLowQuality(props.kbId);
});
</script>

<template>
  <div>
    <el-table :data="adminKbStore.lowQualityChunks" border size="small">
      <el-table-column label="片段内容" min-width="240" show-overflow-tooltip>
        <template #default="{ row }">
          {{ (row as LowQualityChunkVO).content }}
        </template>
      </el-table-column>
      <el-table-column label="来源文档" min-width="160" show-overflow-tooltip>
        <template #default="{ row }">
          {{
            (row as LowQualityChunkVO).documentTitle ||
            `#${(row as LowQualityChunkVO).documentId}`
          }}
        </template>
      </el-table-column>
      <el-table-column label="分块序号" width="90" align="center">
        <template #default="{ row }">
          {{ (row as LowQualityChunkVO).chunkIndex ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="点踩次数" width="100" align="center">
        <template #default="{ row }">
          <el-tag type="danger" size="small">
            {{ (row as LowQualityChunkVO).thumbsDownCount }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="180" align="center">
        <template #default="{ row }">
          <el-button
            v-has-perm="['kb:audit']"
            size="small"
            link
            type="danger"
            :loading="disposingId === (row as LowQualityChunkVO).chunkId"
            @click="handleDispose(row as LowQualityChunkVO, 'clean')"
          >
            清理
          </el-button>
          <el-button
            v-has-perm="['kb:audit']"
            size="small"
            link
            type="warning"
            @click="handleDispose(row as LowQualityChunkVO, 'rechunk')"
          >
            重新分块
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="flex justify-end mt-4">
      <el-pagination
        :current-page="adminKbStore.lowQualityQuery.pageNum"
        :page-size="adminKbStore.lowQualityQuery.pageSize"
        :total="adminKbStore.lowQualityTotal"
        :page-sizes="[10, 20, 50]"
        background
        layout="total, sizes, prev, pager, next"
        @current-change="handlePageChange"
        @size-change="handleSizeChange"
      />
    </div>
  </div>
</template>
