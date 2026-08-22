import request from "@/utils/request";
import { createWebSocket, type WSClient } from "@/utils/websocket";
import {
  AsrResultVO,
  AsrStreamHandlers,
  AsrStreamMessage,
  HotwordForm,
  HotwordVO,
  OfflineAsrForm,
  OfflineAsrResultVO,
  StreamAsrSessionForm,
  StreamAsrSessionVO,
  TtsForm,
  TtsResultVO,
  VoiceVO,
} from "./model";

/** 流式 ASR 会话句柄，封装 WebSocket 连接与音频发送 */
export interface AsrStreamSession {
  /** WebSocket 客户端，用于直接操作（如检查连接状态） */
  ws: WSClient;
  /** 发送 PCM 音频块（16kHz, 16bit, mono） */
  sendAudio: (chunk: ArrayBuffer | ArrayBufferView) => void;
  /**
   * 发送 EOS 结束信号，告知服务端音频已结束。
   *
   * 发送后等待服务端推送最终识别结果并关闭连接，不要立即调用 close()。
   */
  stop: () => void;
  /** 主动关闭连接（通常无需手动调用，服务端会在 EOS 后自动关闭） */
  close: () => void;
}

class VoiceAPI {
  /**
   * 创建并启动流式 ASR 会话。
   *
   * 封装完整流程：
   * 1. 调用 `createStreamAsrSession` 获取 WebSocket 地址
   * 2. 建立 WebSocket 连接
   * 3. 返回会话句柄，调用方通过 `sendAudio` 推送 PCM 音频块，
   *    通过 `stop` 发送结束信号
   *
   * @param form 会话创建参数
   * @param handlers 识别结果回调
   * @returns 流式 ASR 会话句柄
   */
  static async startStreamAsr(
    form: StreamAsrSessionForm,
    handlers: AsrStreamHandlers
  ): Promise<AsrStreamSession> {
    const session = await this.createStreamAsrSession(form);

    const ws = createWebSocket({
      url: session.wsUrl,
      handlers: {
        onMessage: (data: string) => {
          try {
            const msg = JSON.parse(data) as AsrStreamMessage;
            handlers.onMessage(msg);
          } catch {
            // 非 JSON 消息忽略
          }
        },
        ...(handlers.onOpen ? { onOpen: handlers.onOpen } : {}),
        ...(handlers.onClose ? { onClose: handlers.onClose } : {}),
        ...(handlers.onError ? { onError: handlers.onError } : {}),
        ...(handlers.onReconnect ? { onReconnect: handlers.onReconnect } : {}),
      },
    });
    ws.connect();

    return {
      ws,
      sendAudio: (chunk: ArrayBuffer | ArrayBufferView) => {
        ws.sendBinary(chunk);
      },
      stop: () => {
        ws.send("EOS");
      },
      close: () => {
        ws.close();
      },
    };
  }

  /** 创建流式 ASR 会话（获取 WebSocket 连接信息） */
  static createStreamAsrSession(data: StreamAsrSessionForm) {
    return request<StreamAsrSessionVO>({
      url: "/api/v1/voice/asr/stream-session",
      method: "post",
      data,
    });
  }

  /** 离线 ASR 识别（multipart 直传音频文件） */
  static offlineAsr(data: OfflineAsrForm) {
    const formData = new FormData();
    formData.append("file", data.file);
    if (data.model) {
      formData.append("model", data.model);
    }
    return request<OfflineAsrResultVO>({
      url: "/api/v1/voice/asr/offline",
      method: "post",
      data: formData,
      headers: { "Content-Type": "multipart/form-data" },
    });
  }

  /** 查询流式 ASR 会话的最终识别结果 */
  static getAsrResult(sessionId: string) {
    return request<AsrResultVO>({
      url: `/api/v1/voice/asr/result/${sessionId}`,
      method: "get",
    });
  }

  /** 文本转语音（返回音频 URL 或 HTTP 流式响应） */
  static tts(data: TtsForm) {
    return request<TtsResultVO>({
      url: "/api/v1/voice/tts",
      method: "post",
      data,
    });
  }

  /** 获取可用音色列表 */
  static getVoices() {
    return request<VoiceVO[]>({
      url: "/api/v1/voice/tts/voices",
      method: "get",
    });
  }

  // ===== 热词管理 =====

  /** 查询当前用户热词列表 */
  static getHotwords() {
    return request<HotwordVO[]>({
      url: "/api/v1/voice/hotwords",
      method: "get",
    });
  }

  /** 新增用户热词 */
  static addHotword(data: HotwordForm) {
    return request<HotwordVO>({
      url: "/api/v1/voice/hotwords",
      method: "post",
      data,
    });
  }

  /** 删除用户热词 */
  static deleteHotword(id: number) {
    return request({
      url: `/api/v1/voice/hotwords/${id}`,
      method: "delete",
    });
  }

  /** 查询全局热词列表 */
  static getGlobalHotwords() {
    return request<HotwordVO[]>({
      url: "/api/v1/voice/hotwords/global",
      method: "get",
    });
  }

  /** 新增全局热词（管理员） */
  static addGlobalHotword(data: HotwordForm) {
    return request<HotwordVO>({
      url: "/api/v1/voice/hotwords/global",
      method: "post",
      data,
    });
  }

  /** 删除全局热词（管理员） */
  static deleteGlobalHotword(id: number) {
    return request({
      url: `/api/v1/voice/hotwords/global/${id}`,
      method: "delete",
    });
  }
}

export default VoiceAPI;
