<template></template>

<script lang="ts" setup>
import { ref, onUnmounted } from "vue";

// 内部组件：麦克风采集 + PCM 转码，无 UI，由 VoiceInput 编排
defineOptions({ name: "VoiceRecorder" });

const emit = defineEmits<{
  (e: "chunk", chunk: ArrayBuffer): void;
  (e: "stopped"): void;
  (e: "canceled"): void;
}>();

// 识别按秒计费，60s 上限防止失控录音产生高额扣费（需求规格 §3.1.4）
const MAX_DURATION_MS = 60 * 1000;
// 后端识别输入要求：16kHz/16bit/单声道 PCM
const TARGET_SAMPLE_RATE = 16000;

const recording = ref(false);

let mediaStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let processorNode: ScriptProcessorNode | null = null;
let silentGain: GainNode | null = null;
let stopTimer: ReturnType<typeof setTimeout> | null = null;
// 保留完整 PCM，供流式会话失败时编码 WAV 走离线识别兜底
let pcmChunks: Int16Array[] = [];

function downsample(
  input: Float32Array,
  fromRate: number,
  toRate: number
): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += input[j];
    }
    output[i] = sum / (end - start || 1);
  }
  return output;
}

function floatTo16Bit(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

/** 开始录音，权限申请失败时抛出错误由调用方处理 */
async function start() {
  if (recording.value) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true },
    });
  } catch {
    throw new Error("麦克风授权失败，请在浏览器或系统设置中开启麦克风权限");
  }
  pcmChunks = [];
  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    if (!recording.value) return;
    const downsampled = downsample(
      event.inputBuffer.getChannelData(0),
      audioContext!.sampleRate,
      TARGET_SAMPLE_RATE
    );
    const pcm = floatTo16Bit(downsampled);
    pcmChunks.push(pcm);
    emit("chunk", pcm.buffer as ArrayBuffer);
  };
  // 静音增益落地面：ScriptProcessor 需连接到 destination 才会驱动，静音避免回声
  silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  sourceNode.connect(processorNode);
  processorNode.connect(silentGain);
  silentGain.connect(audioContext.destination);
  recording.value = true;
  stopTimer = setTimeout(() => stop(), MAX_DURATION_MS);
}

/** 停止录音并发送结束信号（由调用方触发 EOS） */
function stop() {
  if (!recording.value) return;
  teardown();
  emit("stopped");
}

/** 取消录音：丢弃已采集音频，不产生识别请求 */
function cancel() {
  if (!recording.value) return;
  pcmChunks = [];
  teardown();
  emit("canceled");
}

/** 录音完整音频（WAV 16kHz/16bit/mono），用于流式失败时的离线识别 */
function getWavBlob(): Blob {
  return new Blob([encodeWav(pcmChunks)], { type: "audio/wav" });
}

function teardown() {
  if (stopTimer) {
    clearTimeout(stopTimer);
    stopTimer = null;
  }
  processorNode?.disconnect();
  silentGain?.disconnect();
  sourceNode?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  processorNode = null;
  silentGain = null;
  sourceNode = null;
  mediaStream = null;
  audioContext = null;
  recording.value = false;
}

function encodeWav(chunks: Int16Array[]): ArrayBuffer {
  const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, TARGET_SAMPLE_RATE, true);
  view.setUint32(28, TARGET_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, sampleCount * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++, offset += 2) {
      view.setInt16(offset, chunk[i], true);
    }
  }
  return buffer;
}

onUnmounted(() => {
  if (recording.value) {
    cancel();
  }
});

defineExpose({ start, stop, cancel, getWavBlob });
</script>
