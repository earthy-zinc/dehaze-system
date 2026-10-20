<!-- AI 知识库管理端列表页：公共知识库管理 + 私有库只读监控 -->
<script lang="ts" setup>
import { AiKnowledgeBaseAPI, KnowledgeBaseVO } from "clarity-sdk-js";
import { onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { Plus, Refresh } from "@element-plus/icons-vue";
import { useAdminKbStore } from "@/store/modules/adminKb";
import { useKbDataStore } from "@/store/modules/kbData";
import { formatTokens } from "@/utils/format";

defineOptions({ name: "AiKbList" });

const router = useRouter();
const kbDataStore = useKbDataStore();
const adminKbStore = useAdminKbStore();

// 新建/编辑弹窗：编辑时 embedding 与分块配置由后端限制不可修改，仅名称/描述/检索项生效
const kbFormDialog = reactive({
  visible: false,
  mode: "create" as "create" | "edit",
  kbId: 0,
  form: {
    name: "",
    description: "",
    visibility: "public" as KnowledgeBaseVO["visibility"],
    embeddingModel: "",
    chunkingStrategy: "fixed" as KnowledgeBaseVO["chunkingStrategy"],
    chunkSize: 512,
    chunkOverlap: 64,
    searchStrategy: "vector" as KnowledgeBaseVO["searchStrategy"],
    hybridWeight: 0.7,
    topK: 5,
    scoreThreshold: 0.5,
    enableRerank: false,
    rerankModel: "",
  },
});

function openCreateDialog() {
  kbFormDialog.mode = "create";
  kbFormDialog.kbId = 0;
  kbFormDialog.form = {
    ...kbFormDialog.form,
    name: "",
    description: "",
    visibility: "public",
    searchStrategy: "vector",
    hybridWeight: 0.7,
    topK: 5,
    scoreThreshold: 0.5,
    enableRerank: false,
    rerankModel: "",
  };
  kbFormDialog.visible = true;
}

function openEditDialog(kb: KnowledgeBaseVO) {
  kbFormDialog.mode = "edit";
  kbFormDialog.kbId = kb.id;
  kbFormDialog.form = {
    name: kb.name,
    description: kb.description ?? "",
    visibility: kb.visibility,
    embeddingModel: kb.embeddingModel,
    chunkingStrategy: kb.chunkingStrategy,
    chunkSize: kb.chunkSize,
    chunkOverlap: kb.chunkOverlap,
    searchStrategy: kb.searchStrategy,
    hybridWeight: kb.hybridWeight,
    topK: kb.topK,
    scoreThreshold: kb.scoreThreshold,
    enableRerank: kb.enableRerank === 1,
    rerankModel: kb.rerankModel ?? "",
  };
  kbFormDialog.visible = true;
}

async function submitKbForm(form: typeof kbFormDialog.form) {
  if (kbFormDialog.mode === "create") {
    await AiKnowledgeBaseAPI.create(form);
    ElMessage.success("知识库创建成功");
  } else {
    await AiKnowledgeBaseAPI.update(kbFormDialog.kbId, {
      name: form.name,
      description: form.description,
      searchStrategy: form.searchStrategy,
      hybridWeight: form.hybridWeight,
      topK: form.topK,
      scoreThreshold: form.scoreThreshold,
      enableRerank: form.enableRerank,
      rerankModel: form.rerankModel,
    });
    ElMessage.success("知识库配置已更新");
  }
  kbFormDialog.visible = false;
  await refreshList();
}

function handleCardClick(kb: KnowledgeBaseVO) {
  router.push(`/admin/ai-knowledge/${kb.id}`);
}

async function handleCardDelete(kb: KnowledgeBaseVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除知识库「${kb.name}」？将同步删除其 ES 索引，不可恢复。`,
      "删除确认",
      { type: "warning", confirmButtonText: "确定", cancelButtonText: "取消" }
    );
  } catch {
    return;
  }
  await AiKnowledgeBaseAPI.delete(kb.id);
  ElMessage.success("知识库已删除");
  await refreshList();
}

async function refreshList() {
  await kbDataStore.fetchKbList();
  if (adminKbStore.adminTab === "private") {
    await adminKbStore.fetchPrivateKbs();
  }
}

function handleTabChange(tab: "public" | "private") {
  adminKbStore.switchAdminTab(tab);
}

onMounted(() => {
  kbDataStore.initScope("admin");
  kbDataStore.fetchKbList();
});
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <KbAdminTabs
        :model-value="adminKbStore.adminTab"
        @update:model-value="handleTabChange"
      />
    </div>

    <el-card
      v-show="adminKbStore.adminTab === 'public'"
      shadow="never"
      class="!border-none"
    >
      <div class="flex justify-between mb-4">
        <span class="font-bold">公共知识库管理</span>
        <div>
          <el-button
            v-has-perm="['kb:manage']"
            type="success"
            @click="openCreateDialog"
          >
            <el-icon><Plus /></el-icon>
            新建知识库
          </el-button>
          <el-button @click="refreshList">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
      <!-- 与「私有库监控」统一为表格：同一页面两个 Tab 布局语言一致，切换不跳戏 -->
      <el-table
        v-loading="kbDataStore.loading"
        :data="kbDataStore.kbList"
        size="small"
        @row-click="handleCardClick"
      >
        <el-table-column label="知识库名称" min-width="160">
          <template #default="{ row }">
            <span class="font-medium">{{ row.name }}</span>
          </template>
        </el-table-column>
        <el-table-column label="描述" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">{{ row.description || "—" }}</template>
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
        <el-table-column label="Token" width="100" align="center">
          <template #default="{ row }">{{
            formatTokens(row.totalTokens)
          }}</template>
        </el-table-column>
        <el-table-column label="操作" width="140" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-has-perm="['kb:manage']"
              link
              type="primary"
              size="small"
              @click.stop="openEditDialog(row as KnowledgeBaseVO)"
            >
              编辑
            </el-button>
            <el-button
              v-has-perm="['kb:manage']"
              link
              type="danger"
              size="small"
              @click.stop="handleCardDelete(row as KnowledgeBaseVO)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无公共知识库" :image-size="60" />
        </template>
      </el-table>
    </el-card>

    <el-card
      v-show="adminKbStore.adminTab === 'private'"
      shadow="never"
      class="!border-none"
    >
      <div class="mb-4 font-bold">私有库监控（只读）</div>
      <PrivateKbMonitorTable
        :kbs="adminKbStore.privateKbs"
        :loading="adminKbStore.privateLoading"
      />
    </el-card>

    <el-dialog
      v-model="kbFormDialog.visible"
      :title="kbFormDialog.mode === 'create' ? '新建知识库' : '编辑知识库'"
      width="640px"
      append-to-body
    >
      <KbConfigForm
        v-model="kbFormDialog.form"
        :mode="kbFormDialog.mode"
        scope="admin"
        @submit="submitKbForm"
      />
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.search-container {
  padding: 16px;
  margin-bottom: 16px;
  background: #fff;
  border-radius: 4px;
}
</style>
