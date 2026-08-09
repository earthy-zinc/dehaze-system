/** 流式 ASR 会话创建请求 */
export interface StreamAsrSessionForm {
  /** ASR 模型（默认流式模型 sensevoice） */
  model?: string;
}

/** 流式 ASR 会话创建结果 */
export interface StreamAsrSessionVO {
  /** 会话 ID，用于查询识别结果与绑定 WebSocket */
  sessionId: string;
  /** WebSocket 连接地址（业务后端 /ws/asr） */
  wsUrl: string;
}

/** 离线 ASR 识别请求 */
export interface OfflineAsrForm {
  /** 音频文件（multipart 直传，WAV/PCM，16kHz/16bit/mono） */
  file: File | Blob;
  /** ASR 模型（默认离线高精度模型 paraformer） */
  model?: string;
}

/** 热词 */
export interface HotwordVO {
  /** 热词 ID */
  id: number;
  /** 热词内容 */
  word: string;
  /** 创建时间 */
  createTime?: string;
}

/** 热词新增请求 */
export interface HotwordForm {
  /** 热词内容 */
  word: string;
}

/** 离线 ASR 识别结果 */
export interface OfflineAsrResultVO {
  /** 会话 ID */
  sessionId: string;
  /** 完整识别文本（含标点断句） */
  text: string;
}

/** 流式 ASR 会话最终识别结果 */
export interface AsrResultVO {
  /** 会话 ID */
  sessionId: string;
  /** 最终识别文本 */
  text: string;
  /** 识别状态 */
  status: "completed" | "processing" | "failed";
}

/** TTS 合成请求 */
export interface TtsForm {
  /** 待合成文本 */
  text: string;
  /** 音色 */
  voice?: string;
  /** 语速（默认 1.0） */
  speed?: number;
  /** 音频格式（默认 mp3） */
  format?: string;
  /** 采样率（默认 16000） */
  sampleRate?: number;
}

/** TTS 合成结果 */
export interface TtsResultVO {
  /** 音频 URL（非流式返回） */
  audioUrl?: string;
  /** 音频格式（默认 mp3） */
  format?: string;
}

/** 可用音色 */
export interface VoiceVO {
  /** 音色 ID */
  id: string;
  /** 音色名称 */
  name: string;
  /** 音色描述 */
  description?: string;
  /** 音色语言/风格标签 */
  tags?: string[];
}
