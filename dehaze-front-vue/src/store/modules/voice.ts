// 语音交互 store：录音/识别/播报状态 + 播报偏好 + 用户级热词
import { VoiceAPI, type HotwordVO } from "dehaze-sdk-js";

export type RecordState = "idle" | "recording" | "stopped" | "canceled";
export type PlaybackState = "idle" | "playing" | "stopped";
export type WsStatus = "connecting" | "open" | "closed" | "error";

export interface TtsPreference {
  // 语音播报属主动能力，默认关闭避免打扰
  enabled: boolean;
  voiceId: string;
  speed: number;
}

const TTS_PREFERENCE_KEY = "voice:tts-preference";

function loadTtsPreference(): TtsPreference {
  try {
    const raw = localStorage.getItem(TTS_PREFERENCE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<TtsPreference>;
      return {
        enabled: parsed.enabled === true,
        voiceId: parsed.voiceId ?? "",
        speed: parsed.speed ?? 1,
      };
    }
  } catch {
    // localStorage 不可用时使用默认偏好
  }
  return { enabled: false, voiceId: "", speed: 1 };
}

export const useVoiceStore = defineStore("voice", () => {
  const recordState = ref<RecordState>("idle");
  const recognizedText = ref("");
  const asrSession = reactive({ wsStatus: "closed" as WsStatus });
  const playbackState = ref<PlaybackState>("idle");
  const hotwords = ref<HotwordVO[]>([]);
  const ttsPreference = reactive<TtsPreference>(loadTtsPreference());

  // 播报音频单实例：再次播放前先停止上一段，避免多路音频叠加
  let audio: HTMLAudioElement | null = null;

  // ===== 录音状态 =====

  function startRecording() {
    recordState.value = "recording";
    recognizedText.value = "";
  }

  function stopRecording() {
    recordState.value = "stopped";
  }

  function cancelRecording() {
    recordState.value = "canceled";
    recognizedText.value = "";
  }

  // ===== ASR =====

  /** 离线识别完整音频，返回识别文本（流式会话异常时的兜底路径） */
  async function offlineRecognize(file: File | Blob): Promise<string> {
    const result = await VoiceAPI.offlineAsr({ file });
    return result.text;
  }

  // ===== TTS 播报 =====

  /** 程序化播报入口（如语音回复开关开启后的自动朗读），音色/语速缺省取用户偏好 */
  async function playSpeech(
    text: string,
    options?: { voiceId?: string; speed?: number }
  ) {
    // 空文本防御：TTS 接口 text 必填（min_length=1），空/纯空白文本直接返回不发请求
    if (!text || !text.trim()) {
      return;
    }
    const voiceId = options?.voiceId ?? ttsPreference.voiceId;
    const speed = options?.speed ?? ttsPreference.speed;
    const result = await VoiceAPI.tts({
      text,
      voice: voiceId || undefined,
      speed,
    });
    if (!result.audioUrl) {
      throw new Error("语音合成结果为空");
    }
    playAudio(result.audioUrl);
  }

  function playAudio(url: string) {
    if (audio) {
      audio.pause();
      audio = null;
    }
    audio = new Audio(url);
    audio.onended = () => {
      playbackState.value = "idle";
    };
    audio.onerror = () => {
      playbackState.value = "idle";
      ElMessage.error("语音播放失败");
    };
    playbackState.value = "playing";
    audio.play().catch(() => {
      playbackState.value = "idle";
    });
  }

  function stopSpeech() {
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    playbackState.value = "stopped";
  }

  // ===== 播报偏好 =====

  function updatePreference(partial: Partial<TtsPreference>) {
    Object.assign(ttsPreference, partial);
    localStorage.setItem(
      TTS_PREFERENCE_KEY,
      JSON.stringify({
        enabled: ttsPreference.enabled,
        voiceId: ttsPreference.voiceId,
        speed: ttsPreference.speed,
      })
    );
  }

  // ===== 用户级热词 =====

  async function fetchHotwords() {
    hotwords.value = await VoiceAPI.getHotwords();
  }

  async function addHotword(word: string) {
    const hotword = await VoiceAPI.addHotword({ word });
    hotwords.value.push(hotword);
  }

  async function deleteHotword(id: number) {
    await VoiceAPI.deleteHotword(id);
    hotwords.value = hotwords.value.filter((item) => item.id !== id);
  }

  return {
    recordState,
    recognizedText,
    asrSession,
    playbackState,
    hotwords,
    ttsPreference,
    startRecording,
    stopRecording,
    cancelRecording,
    offlineRecognize,
    playSpeech,
    playAudio,
    stopSpeech,
    updatePreference,
    fetchHotwords,
    addHotword,
    deleteHotword,
  };
});
