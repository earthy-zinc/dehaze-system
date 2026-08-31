<!-- 创建向导弹窗：嵌入 KbConfigForm（推荐默认值 + 不可改项提示） -->
<script lang="ts" setup>
import type { KnowledgeBaseCreateForm } from "dehaze-sdk-js";
import { ref } from "vue";
import { useUserKbStore } from "@/store/modules/userKb";

defineOptions({ name: "CreateKbGuide" });

defineProps<{
  visible: boolean;
}>();

const emit = defineEmits<{
  (e: "update:visible", value: boolean): void;
}>();

const userKbStore = useUserKbStore();
// v-if 重建保证每次打开重新初始化推荐默认值
const form = ref<Partial<KnowledgeBaseCreateForm>>({ visibility: "private" });

async function handleSubmit(value: Partial<KnowledgeBaseCreateForm>) {
  try {
    await userKbStore.submitCreate(value as KnowledgeBaseCreateForm);
    emit("update:visible", false);
  } catch {
    // 错误已由全局拦截器提示（如配额超限），弹窗保持打开供修改后重试
  }
}
</script>

<template>
  <el-dialog
    :model-value="visible"
    title="创建知识库"
    width="640px"
    @update:model-value="emit('update:visible', $event)"
  >
    <KbConfigForm
      v-if="visible"
      v-model="form"
      mode="create"
      scope="self"
      @submit="handleSubmit"
    />
  </el-dialog>
</template>
