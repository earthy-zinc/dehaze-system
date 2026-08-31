<!-- 分块预览弹窗：基于 fileId + 分块配置预览，不向量化不写索引 -->
<script lang="ts" setup>
import type { ChunkingStrategy } from "dehaze-sdk-js";
import { AiKnowledgeBaseAPI } from "dehaze-sdk-js";
import { reactive, ref, watch } from "vue";

defineOptions({ name: "ChunkPreviewDialog" });

interface ChunkPreviewConfig {
  chunkingStrategy: ChunkingStrategy;
  chunkSize: number;
  chunkOverlap: number;
}

const props = defineProps<{
  visible: boolean;
  fileId?: number;
  chunkConfig: ChunkPreviewConfig;
}>();

const emit = defineEmits<{
  (e: "update:visible", value: boolean): void;
  (e: "confirm", config: ChunkPreviewConfig): void;
}>();

const loading = ref(false);
const chunks = ref<{ index: number; content: string; tokenCount: number }[]>(
  []
);
const editableConfig = reactive<ChunkPreviewConfig>({ ...props.chunkConfig });

watch(
  () => props.visible,
  (visible) => {
    if (!visible) return;
    Object.assign(editableConfig, props.chunkConfig);
    loadPreview();
  }
);

async function loadPreview() {
  if (!props.fileId) return;
  loading.value = true;
  try {
    chunks.value = await AiKnowledgeBaseAPI.previewChunks({
      fileId: props.fileId,
      chunkingStrategy: editableConfig.chunkingStrategy,
      chunkSize: editableConfig.chunkSize,
      chunkOverlap: editableConfig.chunkOverlap,
    });
  } catch {
    // 预览失败已由全局拦截器提示
    chunks.value = [];
  } finally {
    loading.value = false;
  }
}

function handleConfirm() {
  emit("confirm", { ...editableConfig });
  emit("update:visible", false);
}
</script>

<template>
  <el-dialog
    :model-value="visible"
    title="分块预览"
    width="720px"
    append-to-body
    @update:model-value="emit('update:visible', $event)"
  >
    <div class="flex items-end gap-4 mb-4">
      <div>
        <div class="config-label">分块大小（token）</div>
        <el-input-number
          v-model="editableConfig.chunkSize"
          :min="100"
          :max="4000"
          :step="50"
        />
      </div>
      <div>
        <div class="config-label">分块重叠（token）</div>
        <el-input-number
          v-model="editableConfig.chunkOverlap"
          :min="0"
          :max="1000"
          :step="10"
        />
      </div>
      <el-button :loading="loading" @click="loadPreview">重新预览</el-button>
    </div>

    <el-table v-loading="loading" :data="chunks" border max-height="400">
      <el-table-column label="#" prop="index" width="60" align="center" />
      <el-table-column label="分块内容" min-width="360">
        <template #default="{ row }">
          <div class="chunk-content">{{ row.content }}</div>
        </template>
      </el-table-column>
      <el-table-column
        label="Token"
        prop="tokenCount"
        width="90"
        align="center"
      />
    </el-table>
    <div
      v-if="!loading && chunks.length === 0"
      class="text-center py-4 text-gray-400"
    >
      暂无预览数据
    </div>

    <template #footer>
      <el-button @click="emit('update:visible', false)">取消</el-button>
      <el-button type="primary" :disabled="loading" @click="handleConfirm">
        确认入库
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.config-label {
  margin-bottom: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.chunk-content {
  display: -webkit-box;
  max-height: 88px;
  overflow: hidden;
  -webkit-line-clamp: 4;
  font-size: 12px;
  line-height: 1.5;
  word-break: break-all;
  white-space: pre-wrap;
  -webkit-box-orient: vertical;
}
</style>
