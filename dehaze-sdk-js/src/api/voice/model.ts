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

/** 流式 ASR 识别增量文本（后端 WebSocket 下行 JSON） */
export interface AsrStreamMessage {
  /** 识别文本 */
  text: string;
  /** 是否为最终结果 */
  isFinal: boolean;
}

/** 流式 ASR 回调集合 */
export interface AsrStreamHandlers {
  /** 收到识别增量文本时触发 */
  onMessage: (message: AsrStreamMessage) => void;
  /** WebSocket 连接建立时触发（可开始发送音频） */
  onOpen?: () => void;
  /** 连接关闭时触发 */
  onClose?: (code: number, reason: string) => void;
  /** 连接错误时触发 */
  onError?: (error: Event) => void;
  /** 重连尝试时触发（attempt 为第几次重试） */
  onReconnect?: (attempt: number) => void;
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

/** ASR 引擎运行状态（服务监控） */
export interface AsrEngineStatusVO {
  /** 引擎在线状态（online=在线 / offline=离线） */
  engineStatus: "online" | "offline";
  /** 当前并发会话数 */
  concurrentSessions: number;
  /** 最大并发会话数 */
  maxConcurrentSessions: number;
  /** 流式模型加载状态 */
  streamModelLoaded: boolean;
  /** 离线模型加载状态 */
  offlineModelLoaded: boolean;
}

/** TTS 引擎运行状态（服务监控） */
export interface TtsEngineStatusVO {
  /** 引擎在线状态（online=在线 / offline=离线） */
  engineStatus: "online" | "offline";
  /** 当前音色模型加载状态 */
  voiceModelLoaded: boolean;
}

/** 语音服务状态（管理端服务监控，GET /api/v1/voice/service/status） */
export interface ServiceStatusVO {
  /** ASR 引擎状态 */
  asr: AsrEngineStatusVO;
  /** TTS 引擎状态 */
  tts: TtsEngineStatusVO;
}
