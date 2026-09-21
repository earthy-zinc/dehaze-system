<!-- 产物详情只读弹窗：用户端对话页与管理端会话审计抽屉共用。
     点击产物卡片 → 按需拉取 getArtifactDetail → 只读展示（无任何内容操作入口）。
     拉取与弹窗收敛在视图层，ai-chat 组件层保持零 SDK。 -->
<script lang="ts" setup>
import { AiConversationAPI } from "dehaze-sdk-js";
import { ref } from "vue";
import type { ChatArtifactVM } from "@/components/ai-chat/types";

defineOptions({ name: "ArtifactDetailDialog" });

const visible = ref(false);
const detail = ref<Record<string, unknown> | null>(null);

/** 打开产物详情：拉取详情后弹出（失败静默，不打断消息浏览） */
async function open(artifact: ChatArtifactVM) {
  try {
    detail.value = await AiConversationAPI.getArtifactDetail(artifact.id);
    visible.value = true;
  } catch {
    // 产物详情拉取失败不打断浏览
  }
}

defineExpose({ open });
</script>

<template>
  <el-dialog v-model="visible" title="产物详情" width="640px">
    <pre v-if="detail" class="artifact-detail-dialog__body">
        {{ JSON.stringify(detail, null, 2) }}
    </pre>
  </el-dialog>
</template>

<style scoped lang="scss">
.artifact-detail-dialog__body {
  max-height: 420px;
  overflow: auto;
  font-size: 12px;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
</style>
