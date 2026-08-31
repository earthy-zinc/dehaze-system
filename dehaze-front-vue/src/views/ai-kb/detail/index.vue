<!-- AI 知识库管理端详情页：配置/文档/索引/检索质量四区（私有库不进入详情页） -->
<script lang="ts" setup>
import { AiKnowledgeBaseAPI, DocumentVO } from "dehaze-sdk-js";
import { onMounted, reactive } from "vue";
import { useRoute } from "vue-router";
import { useAdminKbStore } from "@/store/modules/adminKb";
import { useKbDataStore } from "@/store/modules/kbData";

defineOptions({ name: "AiKbDetail" });

const route = useRoute();
const kbId = Number(route.params.id);
const kbDataStore = useKbDataStore();
const adminKbStore = useAdminKbStore();

// 文档查看弹窗（DocumentTable 的 view 事件）
const docDialog = reactive({
  visible: false,
  title: "",
  content: "",
});

async function refreshDocuments(resetPage = false) {
  if (resetPage) {
    kbDataStore.documentQuery.pageNum = 1;
  }
  await kbDataStore.fetchDocuments(kbId);
}

// DocumentTable 分页状态在组件内部维护，需同步写回 documentQuery 再拉取
async function handlePageChange(page: { pageNum: number; pageSize: number }) {
  kbDataStore.documentQuery.pageNum = page.pageNum;
  kbDataStore.documentQuery.pageSize = page.pageSize;
  await kbDataStore.fetchDocuments(kbId);
}

async function handleDocView(doc: DocumentVO) {
  const detail = await AiKnowledgeBaseAPI.getDocumentDetail(doc.id);
  docDialog.title = detail.title;
  docDialog.content = detail.content ?? detail.rawContent ?? "（无解析内容）";
  docDialog.visible = true;
}

async function handleDocDelete(doc: DocumentVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除文档「${doc.title}」及其关联分块？`,
      "删除确认",
      { type: "warning", confirmButtonText: "确定", cancelButtonText: "取消" }
    );
  } catch {
    return;
  }
  await AiKnowledgeBaseAPI.deleteDocument(doc.id);
  ElMessage.success("文档已删除");
  refreshDocuments(true);
}

async function handleDocReprocess(doc: DocumentVO) {
  await AiKnowledgeBaseAPI.reprocessDocument(doc.id);
  ElMessage.success("已提交重新处理，完成后自动重建索引");
  refreshDocuments(true);
}

// 上传入库后刷新文档列表并同步知识库统计
async function handleUploaded() {
  await refreshDocuments(true);
  kbDataStore.fetchKbDetail(kbId);
}

onMounted(async () => {
  kbDataStore.initScope("admin");
  await Promise.all([
    kbDataStore.fetchKbDetail(kbId),
    kbDataStore.fetchDocuments(kbId),
  ]);
  adminKbStore.fetchIndexStats(kbId);
});
</script>

<template>
  <div class="app-container">
    <KbDetailHeader :kb="kbDataStore.kbDetail" />

    <div class="mt-4">
      <KbConfigPanel :kb="kbDataStore.kbDetail" />
    </div>

    <el-card shadow="never" class="!border-none mt-4">
      <template #header>
        <span>文档管理</span>
      </template>
      <div class="mb-4">
        <DocumentUploadPanel
          :knowledge-base-id="kbId"
          @uploaded="handleUploaded"
        />
      </div>
      <DocumentTable
        :documents="kbDataStore.documents"
        :total="kbDataStore.total"
        :loading="kbDataStore.loading"
        :readonly="false"
        @view="handleDocView"
        @delete="handleDocDelete"
        @reprocess="handleDocReprocess"
        @page-change="handlePageChange"
      />
    </el-card>

    <div class="mt-4">
      <VersionHistoryPanel
        :documents="kbDataStore.documents"
        @restored="refreshDocuments(true)"
      />
    </div>

    <div class="mt-4">
      <IndexStatusPanel :kb-id="kbId" />
    </div>

    <div class="mt-4">
      <QualitySection :kb-id="kbId" />
    </div>

    <el-dialog
      v-model="docDialog.visible"
      :title="docDialog.title"
      width="640px"
      append-to-body
    >
      <div class="whitespace-pre-wrap break-words text-sm leading-6">
        {{ docDialog.content }}
      </div>
    </el-dialog>
  </div>
</template>
