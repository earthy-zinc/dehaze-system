<!-- 版本历史：最近 5 个版本列表，查看历史版本内容与恢复 -->
<script lang="ts" setup>
import { AiKnowledgeBaseAPI, DocumentVO } from "dehaze-sdk-js";
import { computed, ref } from "vue";

defineOptions({ name: "VersionHistoryPanel" });

const props = defineProps<{
  documents: DocumentVO[];
}>();

const emit = defineEmits<{
  (e: "restored"): void;
}>();

// 最近 5 个版本：按更新时间倒序取 5 条
const recentVersions = computed(() =>
  [...props.documents]
    .sort((a, b) =>
      (b.updateTime ?? b.createTime).localeCompare(a.updateTime ?? a.createTime)
    )
    .slice(0, 5)
);

const contentDialog = reactive({
  visible: false,
  title: "",
  content: "",
});
const restoringId = ref<number | null>(null);

async function viewContent(doc: DocumentVO) {
  const detail = await AiKnowledgeBaseAPI.getDocumentDetail(doc.id);
  contentDialog.title = `${detail.title}（v${detail.version}）`;
  contentDialog.content =
    detail.content ?? detail.rawContent ?? "（无解析内容）";
  contentDialog.visible = true;
}

/**
 * 恢复历史版本：版本恢复专有接口后端规划中，
 * 暂以重新处理文档兜底（重跑解析与分块流水线并重建索引）。
 */
async function handleRestore(doc: DocumentVO) {
  try {
    await ElMessageBox.confirm(
      `确认将「${doc.title}」恢复至 v${doc.version}？恢复后将重新解析并重建索引。`,
      "恢复确认",
      {
        type: "warning",
        confirmButtonText: "确认恢复",
        cancelButtonText: "取消",
      }
    );
  } catch {
    return;
  }
  restoringId.value = doc.id;
  try {
    await AiKnowledgeBaseAPI.reprocessDocument(doc.id);
    ElMessage.success("已提交恢复，重建索引完成后生效");
    emit("restored");
  } finally {
    restoringId.value = null;
  }
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <template #header>
      <span>版本历史（最近 5 个版本）</span>
    </template>
    <el-table :data="recentVersions" border size="small">
      <el-table-column
        label="文档标题"
        prop="title"
        min-width="180"
        show-overflow-tooltip
      />
      <el-table-column label="版本" prop="version" width="80" align="center">
        <template #default="{ row }">
          <el-tag size="small">v{{ (row as DocumentVO).version }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="更新时间" width="170" align="center">
        <template #default="{ row }">
          <span>{{
            (row as DocumentVO).updateTime || (row as DocumentVO).createTime
          }}</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="180" align="center">
        <template #default="{ row }">
          <el-button
            size="small"
            link
            type="primary"
            @click="viewContent(row as DocumentVO)"
          >
            查看内容
          </el-button>
          <el-button
            v-has-perm="['kb:document:manage']"
            size="small"
            link
            type="warning"
            :loading="restoringId === (row as DocumentVO).id"
            @click="handleRestore(row as DocumentVO)"
          >
            恢复
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog
      v-model="contentDialog.visible"
      :title="contentDialog.title"
      width="640px"
      append-to-body
    >
      <div class="whitespace-pre-wrap break-words text-sm leading-6">
        {{ contentDialog.content }}
      </div>
    </el-dialog>
  </el-card>
</template>
