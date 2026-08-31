<!-- 文档管理区：上传面板 + 文档列表，处理中状态轮询刷新 -->
<script lang="ts" setup>
import type { DocumentVO, KnowledgeBaseVO } from "dehaze-sdk-js";
import { AiKnowledgeBaseAPI } from "dehaze-sdk-js";
import { ElMessage, ElMessageBox } from "element-plus";
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useKbDataStore } from "@/store/modules/kbData";

defineOptions({ name: "DocumentSection" });

const props = defineProps<{
  kb: KnowledgeBaseVO;
}>();

const kbDataStore = useKbDataStore();

const isPublic = computed(() => props.kb.visibility === "public");
const chunkConfig = computed(() => ({
  chunkingStrategy: props.kb.chunkingStrategy,
  chunkSize: props.kb.chunkSize,
  chunkOverlap: props.kb.chunkOverlap,
}));

// 后端 ws 端点未提供，处理中状态以列表接口 5 秒轮询兜底
let pollTimer: ReturnType<typeof setInterval> | null = null;

const detailDrawer = ref(false);
const detailDoc = ref<DocumentVO | null>(null);
const detailLoading = ref(false);

async function refresh() {
  await kbDataStore.fetchDocuments(props.kb.id);
  syncPolling();
}

function syncPolling() {
  const hasActive = kbDataStore.documents.some(
    (doc) =>
      doc.processingStatus === "processing" ||
      doc.processingStatus === "pending"
  );
  if (hasActive && !pollTimer) {
    pollTimer = setInterval(refresh, 5000);
  } else if (!hasActive && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function handlePageChange(query: { pageNum: number; pageSize: number }) {
  kbDataStore.documentQuery.pageNum = query.pageNum;
  kbDataStore.documentQuery.pageSize = query.pageSize;
  refresh();
}

async function handleView(doc: DocumentVO) {
  detailLoading.value = true;
  detailDrawer.value = true;
  try {
    detailDoc.value = await AiKnowledgeBaseAPI.getDocumentDetail(doc.id);
  } catch {
    // 错误已由全局拦截器提示
  } finally {
    detailLoading.value = false;
  }
}

async function handleDelete(doc: DocumentVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除文档 "${doc.title}" 及其分块数据？`,
      "删除确认",
      { type: "warning" }
    );
  } catch {
    return;
  }
  await AiKnowledgeBaseAPI.deleteDocument(doc.id);
  ElMessage.success("文档已删除");
  await refresh();
}

async function handleReprocess(doc: DocumentVO) {
  await AiKnowledgeBaseAPI.reprocessDocument(doc.id);
  ElMessage.success("已重新提交处理");
  await refresh();
}

async function handleUploaded() {
  kbDataStore.documentQuery.pageNum = 1;
  await refresh();
}

watch(
  () => props.kb.id,
  () => refresh()
);

onMounted(refresh);

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
});
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <template #header>
      <div class="flex items-center justify-between">
        <span>文档管理</span>
        <el-button link type="primary" @click="refresh">手动刷新</el-button>
      </div>
    </template>

    <DocumentUploadPanel
      v-if="!isPublic"
      :knowledge-base-id="kb.id"
      :chunk-config="chunkConfig"
      class="mb-4"
      @uploaded="handleUploaded"
    />

    <DocumentTable
      :documents="kbDataStore.documents"
      :total="kbDataStore.total"
      :loading="kbDataStore.loading"
      :readonly="isPublic"
      @view="handleView"
      @delete="handleDelete"
      @reprocess="handleReprocess"
      @page-change="handlePageChange"
    />

    <el-drawer v-model="detailDrawer" :title="detailDoc?.title" size="50%">
      <div v-loading="detailLoading">
        <pre v-if="detailDoc?.content" class="doc-content">{{
          detailDoc.content
        }}</pre>
        <el-empty v-else-if="!detailLoading" description="暂无解析内容" />
      </div>
    </el-drawer>
  </el-card>
</template>

<style lang="scss" scoped>
.doc-content {
  margin: 0;
  font-size: 13px;
  line-height: 1.8;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
</style>
