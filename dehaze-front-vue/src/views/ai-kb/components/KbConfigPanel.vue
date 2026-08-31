<!-- 配置管理区：全量配置编辑（公共 KbConfigForm）+ embedding 迁移入口 -->
<script lang="ts" setup>
import { AiKnowledgeBaseAPI, KnowledgeBaseVO } from "dehaze-sdk-js";
import { reactive, watch } from "vue";
import { useKbDataStore } from "@/store/modules/kbData";

defineOptions({ name: "KbConfigPanel" });

const props = defineProps<{
  kb: KnowledgeBaseVO | null;
}>();

const kbDataStore = useKbDataStore();

// 编辑表单仅含可修改项：名称/描述/检索策略（分块策略与 embedding 模型创建后不可改）
const editForm = reactive({
  name: "",
  description: "",
  searchStrategy: "vector" as KnowledgeBaseVO["searchStrategy"],
  hybridWeight: 0.7,
  topK: 5,
  scoreThreshold: 0.5,
  enableRerank: false,
  rerankModel: "",
});

watch(
  () => props.kb,
  (kb) => {
    if (!kb) return;
    editForm.name = kb.name;
    editForm.description = kb.description ?? "";
    editForm.searchStrategy = kb.searchStrategy;
    editForm.hybridWeight = kb.hybridWeight;
    editForm.topK = kb.topK;
    editForm.scoreThreshold = kb.scoreThreshold;
    editForm.enableRerank = kb.enableRerank === 1;
    editForm.rerankModel = kb.rerankModel ?? "";
  },
  { immediate: true }
);

async function handleSubmit(form: typeof editForm) {
  if (!props.kb) return;
  await AiKnowledgeBaseAPI.update(props.kb.id, form);
  ElMessage.success("配置已更新");
  // 同步刷新详情，保证头部配置摘要与统计一致
  kbDataStore.fetchKbDetail(props.kb.id);
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <template #header>
      <span>配置管理</span>
    </template>
    <KbConfigForm
      v-model="editForm"
      mode="edit"
      scope="admin"
      @submit="handleSubmit"
    />
    <EmbeddingMigratePanel :kb="props.kb" />
  </el-card>
</template>
