<!-- 知识库详情（用户端）：基本信息 + 文档管理 + 检索测试 -->
<script lang="ts" setup>
import {
  AiKnowledgeBaseAPI,
  type KnowledgeBaseCreateForm,
  type KnowledgeBaseUpdateForm,
} from "dehaze-sdk-js";
import { ElMessage, ElMessageBox } from "element-plus";
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useKbDataStore } from "@/store/modules/kbData";

defineOptions({
  name: "KbDetail",
  inheritAttrs: false,
});

const route = useRoute();
const router = useRouter();
const kbDataStore = useKbDataStore();

const kbId = computed(() => Number(route.params.id));
const kb = computed(() => kbDataStore.kbDetail);
const isPublic = computed(() => kb.value?.visibility === "public");

const editDialogVisible = ref(false);
const editForm = ref<Partial<KnowledgeBaseCreateForm>>({});

async function loadDetail() {
  try {
    await kbDataStore.fetchKbDetail(kbId.value);
  } catch {
    // 详情加载失败已由全局拦截器提示
  }
}

function openEditDialog() {
  if (!kb.value) return;
  // 仅回填可修改项：分块与向量化模型创建后不可改
  editForm.value = {
    name: kb.value.name,
    description: kb.value.description,
    searchStrategy: kb.value.searchStrategy,
    hybridWeight: kb.value.hybridWeight,
    topK: kb.value.topK,
    scoreThreshold: kb.value.scoreThreshold,
    enableRerank: kb.value.enableRerank === 1,
    rerankModel: kb.value.rerankModel,
  };
  editDialogVisible.value = true;
}

async function handleEditSubmit(value: Partial<KnowledgeBaseCreateForm>) {
  await AiKnowledgeBaseAPI.update(kbId.value, value as KnowledgeBaseUpdateForm);
  ElMessage.success("配置已更新");
  editDialogVisible.value = false;
  await loadDetail();
}

async function handleDeleteKb() {
  if (!kb.value) return;
  try {
    await ElMessageBox.confirm(
      `确认删除知识库 "${kb.value.name}" ？删除后文档分块与 ES 索引同步清除`,
      "删除确认",
      { type: "warning" }
    );
  } catch {
    return;
  }
  await AiKnowledgeBaseAPI.delete(kbId.value);
  ElMessage.success("知识库已删除");
  router.push("/kb");
}

onMounted(async () => {
  await loadDetail();
  // 引用溯源跳转：对话页携带 citation 参数时提示定位，原文高亮由后续版本实现
  if (route.query.citation) {
    ElMessage.info(`已定位引用来源：${route.query.citation}`);
  }
});

watch(kbId, loadDetail);
</script>

<template>
  <div class="app-container">
    <template v-if="kb">
      <PublicKbReadOnlyBanner v-if="isPublic" />

      <el-card shadow="never" class="!border-none mb-4">
        <div v-if="!isPublic" class="mb-3 flex justify-end gap-2">
          <el-button @click="openEditDialog">编辑配置</el-button>
          <el-button type="danger" @click="handleDeleteKb"
            >删除知识库</el-button
          >
        </div>
        <KbDetailHeader :kb="kb" />
      </el-card>

      <DocumentSection :kb="kb" />
      <RetrieveTestSection :kb="kb" />
    </template>
    <div v-else v-loading="true" class="min-h-40"></div>

    <el-dialog v-model="editDialogVisible" title="编辑知识库配置" width="640px">
      <KbConfigForm
        v-if="editDialogVisible"
        v-model="editForm"
        mode="edit"
        scope="self"
        @submit="handleEditSubmit"
      />
    </el-dialog>
  </div>
</template>
