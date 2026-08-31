<!-- Embedding 迁移面板：选择目标模型 → 二次确认 → 提交迁移 -->
<script lang="ts" setup>
import { AiModelAPI, AiModelVO, KnowledgeBaseVO } from "dehaze-sdk-js";
import { onMounted, ref } from "vue";
import { useAdminKbStore } from "@/store/modules/adminKb";

defineOptions({ name: "EmbeddingMigratePanel" });

const props = defineProps<{
  kb: KnowledgeBaseVO | null;
}>();

const adminKbStore = useAdminKbStore();

const embeddingModels = ref<AiModelVO[]>([]);
const targetModelId = ref("");
const submitting = ref(false);

onMounted(async () => {
  embeddingModels.value = await AiModelAPI.listEnabledModels("embedding");
});

async function handleMigrate() {
  if (!props.kb) return;
  if (!targetModelId.value) {
    ElMessage.warning("请选择目标 embedding 模型");
    return;
  }
  const target = embeddingModels.value.find(
    (m) => m.modelId === targetModelId.value
  );
  const targetName = target?.displayName ?? targetModelId.value;
  try {
    await ElMessageBox.confirm(
      `将把「${props.kb.name}」的 embedding 模型由 ${props.kb.embeddingModel} 迁移至 ${targetName}，全部分块将批量重新向量化并重建索引，期间检索效果可能受影响。确认迁移？`,
      "迁移确认",
      {
        type: "warning",
        confirmButtonText: "确认迁移",
        cancelButtonText: "取消",
      }
    );
  } catch {
    return;
  }
  submitting.value = true;
  try {
    await adminKbStore.submitEmbeddingMigrate(props.kb.id, targetModelId.value);
    ElMessage.success("迁移任务已提交，请在索引状态区查看");
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div class="migrate-panel">
    <el-divider />
    <div class="mb-2 font-bold">Embedding 模型迁移</div>
    <div class="mb-3 text-sm text-gray-500">
      当前模型：{{
        props.kb?.embeddingModel || "-"
      }}。迁移后需后台批量重新向量化并重建索引。
    </div>
    <div class="flex items-center gap-3">
      <el-select
        v-model="targetModelId"
        placeholder="选择目标 embedding 模型"
        style="width: 280px"
      >
        <el-option
          v-for="model in embeddingModels"
          :key="model.id"
          :label="model.displayName"
          :value="model.modelId"
          :disabled="model.modelId === props.kb?.embeddingModel"
        />
      </el-select>
      <el-button
        v-has-perm="['kb:manage']"
        type="warning"
        :loading="submitting"
        :disabled="!props.kb"
        @click="handleMigrate"
      >
        提交迁移
      </el-button>
    </div>
  </div>
</template>
