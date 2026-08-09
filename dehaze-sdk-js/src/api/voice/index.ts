import request from "@/utils/request";
import {
  AsrResultVO,
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

class VoiceAPI {
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
