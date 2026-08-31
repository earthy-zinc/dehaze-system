<template>
  <div class="voice-input">
    <div v-if="recording || finishing" class="recognize-panel">
      <span class="recognize-text">{{ recognizedDisplay }}</span>
      <el-button
        v-if="recording && !finishing"
        link
        size="small"
        @click="handleCancel"
      >
        取消
      </el-button>
    </div>
    <el-tooltip content="语音输入" placement="top">
      <el-button
        circle
        :type="recording ? 'danger' : 'default'"
        :icon="Microphone"
        :disabled="disabled || finishing"
        @click="handleToggle"
      />
    </el-tooltip>
  </div>
</template>

<script lang="ts" setup>
import { ref, computed, onUnmounted } from "vue";
import type { AsrStreamMessage } from "dehaze-sdk-js";
import { Microphone } from "@element-plus/icons-vue";
import { useVoiceStore } from "@/store/modules/voice";
import VoiceRecorder from "./VoiceRecorder.vue";
import AsrStreamClient from "./AsrStreamClient.vue";
import AsrOfflineClient from "./AsrOfflineClient.vue";

defineOptions({ name: "VoiceInput" });

const props = defineProps<{
  modelValue: string;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
  (e: "recognizing", value: boolean): void;
}>();

const voiceStore = useVoiceStore();
const recorderRef = ref<InstanceType<typeof VoiceRecorder>>();
const asrRef = ref<InstanceType<typeof AsrStreamClient>>();
const offlineRef = ref<InstanceType<typeof AsrOfflineClient>>();

const finishing = ref(false);
const streamFailed = ref(false);
const recording = computed(() => voiceStore.recordState === "recording");

const recognizedDisplay = computed(() => {
  if (!recording.value) {
    return "正在识别…";
  }
  return voiceStore.recognizedText || "正在聆听，请说话…";
});

// EOS 后等待服务端最终文本的回调
let resolveFinal: ((text: string) => void) | null = null;

// 识别结果非 100% 准确，只回填输入框，由用户确认/编辑后发送
function handleToggle() {
  if (recording.value) {
    handleStop();
  } else {
    handleStart();
  }
}

async function handleStart() {
  voiceStore.startRecording();
  emit("recognizing", true);
  streamFailed.value = false;
  try {
    await recorderRef.value?.start();
    await asrRef.value?.connect();
  } catch (error) {
    voiceStore.cancelRecording();
    emit("recognizing", false);
    ElMessage.warning(
      error instanceof Error ? error.message : "无法开启录音，请检查麦克风权限"
    );
  }
}

function handleStop() {
  if (!recording.value || finishing.value) return;
  finishing.value = true;
  recorderRef.value?.stop();
}

function handleCancel() {
  recorderRef.value?.cancel();
}

function onRecorderChunk(chunk: ArrayBuffer) {
  asrRef.value?.sendAudio(chunk);
}

function onRecorderStopped() {
  if (streamFailed.value) {
    recognizeOffline();
    return;
  }
  asrRef.value?.finish();
  waitForFinal(5000).then(complete);
}

function onRecorderCanceled() {
  // 取消不发 EOS：不产生识别请求，零成本
  asrRef.value?.close();
  voiceStore.cancelRecording();
  emit("recognizing", false);
}

function onAsrMessage(message: AsrStreamMessage) {
  if (message.isFinal && resolveFinal) {
    resolveFinal(message.text);
    resolveFinal = null;
  }
}

function onAsrError(message: string) {
  streamFailed.value = true;
  ElMessage.warning(message);
}

function waitForFinal(timeoutMs: number): Promise<string> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (text: string) => {
      if (!settled) {
        settled = true;
        resolve(text);
      }
    };
    resolveFinal = finish;
    setTimeout(() => finish(voiceStore.recognizedText), timeoutMs);
  });
}

async function recognizeOffline() {
  try {
    const blob = recorderRef.value?.getWavBlob();
    const text =
      blob && blob.size > 44 ? await offlineRef.value?.recognize(blob) : "";
    complete(text ?? "");
  } catch {
    complete(voiceStore.recognizedText);
  }
}

function complete(text: string) {
  asrRef.value?.close();
  if (text.trim()) {
    emit("update:modelValue", text);
  }
  voiceStore.recognizedText = "";
  voiceStore.stopRecording();
  emit("recognizing", false);
  finishing.value = false;
}

onUnmounted(() => {
  if (recording.value) {
    asrRef.value?.close();
    voiceStore.cancelRecording();
  }
});
</script>

<style lang="scss" scoped>
.voice-input {
  display: inline-flex;
  gap: 8px;
  align-items: center;
}

.recognize-panel {
  display: flex;
  gap: 8px;
  align-items: center;
  max-width: 320px;
  padding: 4px 12px;
  font-size: 13px;
  background: var(--el-fill-color-lighter);
  border-radius: 12px;

  .recognize-text {
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--el-text-color-primary);
    white-space: nowrap;
  }
}
</style>
