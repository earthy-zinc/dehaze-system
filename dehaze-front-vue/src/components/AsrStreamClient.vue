<template></template>

<script lang="ts" setup>
import { onUnmounted } from "vue";
import {
  VoiceAPI,
  type AsrStreamMessage,
  type AsrStreamSession,
} from "dehaze-sdk-js";
import { useVoiceStore } from "@/store/modules/voice";

// 内部组件：流式 ASR 会话生命周期，无 UI，由 VoiceInput 编排
defineOptions({ name: "AsrStreamClient" });

const emit = defineEmits<{
  (e: "message", message: AsrStreamMessage): void;
  (e: "error", message: string): void;
}>();

const voiceStore = useVoiceStore();

let session: AsrStreamSession | null = null;
// WebSocket 握手完成前到达的音频块先缓冲，避免丢掉开头语音
let pendingChunks: ArrayBuffer[] = [];

/** 创建流式识别会话；创建失败/连接异常时提示降级，不中断录音 */
async function connect() {
  voiceStore.asrSession.wsStatus = "connecting";
  try {
    session = await VoiceAPI.startStreamAsr(
      {},
      {
        onMessage: (message) => {
          emit("message", message);
          voiceStore.recognizedText = message.text;
        },
        onOpen: () => {
          voiceStore.asrSession.wsStatus = "open";
          pendingChunks.forEach((chunk) => session?.sendAudio(chunk));
          pendingChunks = [];
        },
        onError: () => {
          voiceStore.asrSession.wsStatus = "error";
          pendingChunks = [];
          emit("error", "识别连接异常，本次录音将改用离线识别");
        },
        onClose: () => {
          voiceStore.asrSession.wsStatus = "closed";
        },
      }
    );
  } catch {
    // 会话创建失败（并发上限/配额不足等）不阻塞录音，停止时走离线识别兜底
    voiceStore.asrSession.wsStatus = "error";
    emit("error", "识别会话创建失败，本次录音将改用离线识别");
  }
}

function sendAudio(chunk: ArrayBuffer) {
  if (session?.ws.isOpen()) {
    session.sendAudio(chunk);
  } else {
    pendingChunks.push(chunk);
  }
}

/** 发送 EOS 结束信号，等待服务端推送最终识别文本 */
function finish() {
  session?.stop();
}

function close() {
  pendingChunks = [];
  session?.close();
  session = null;
  voiceStore.asrSession.wsStatus = "closed";
}

onUnmounted(() => {
  close();
});

defineExpose({ connect, sendAudio, finish, close });
</script>
