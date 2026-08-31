<!-- 检索测试区容器：验证本人库对话可用性 -->
<script lang="ts" setup>
import type { KnowledgeBaseVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "RetrieveTestSection" });

const props = defineProps<{
  kb: KnowledgeBaseVO;
}>();

const isPublic = computed(() => props.kb.visibility === "public");
</script>

<template>
  <el-card shadow="never" class="!border-none mt-4">
    <template #header>
      <span>检索测试</span>
    </template>
    <RetrieveTestPanel :knowledge-base-id="kb.id" :readonly="isPublic" />
  </el-card>
</template>
