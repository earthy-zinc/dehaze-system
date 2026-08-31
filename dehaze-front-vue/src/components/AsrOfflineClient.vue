<template></template>

<script lang="ts" setup>
import { useVoiceStore } from "@/store/modules/voice";

// 内部组件：离线 ASR，无 UI，由 VoiceInput 在流式会话异常时兜底调用
defineOptions({ name: "AsrOfflineClient" });

const voiceStore = useVoiceStore();

/** 提交完整音频离线识别，返回识别文本；失败向上抛出由调用方降级 */
async function recognize(file: File | Blob): Promise<string> {
  return voiceStore.offlineRecognize(file);
}

defineExpose({ recognize });
</script>
