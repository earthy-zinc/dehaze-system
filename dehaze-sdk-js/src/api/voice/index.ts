import request from "@/utils/request";
import { createWebSocket, type WSClient } from "@/utils/websocket";
import { PageResult } from "@/types";
import {
  AsrResultVO,
  AsrStreamHandlers,
  AsrStreamMessage,
  HotwordForm,
  HotwordVO,
  OfflineAsrForm,
  OfflineAsrResultVO,
  ServiceStatusVO,
  StreamAsrSessionForm,
  StreamAsrSessionVO,
  TtsForm,
  TtsResultVO,
  VoiceModelCreateForm,
  VoiceModelPageQuery,
  VoiceModelUpdateForm,
  VoiceModelVO,
  VoiceProviderCreateForm,
  VoiceProviderKeyCreateForm,
  VoiceProviderKeyUpdateForm,
  VoiceProviderKeyVO,
  VoiceProviderPageQuery,
  VoiceProviderTestResultVO,
  VoiceProviderUpdateForm,
  VoiceProviderVO,
  VoiceVO,
} from "./model";

/** 流式 ASR 会话句柄，封装 WebSocket 连接与音频发送 */
export interface AsrStreamSession {
  /** 会话 ID，用于 `getAsrResult` 兜底查询最终识别结果 */
  sessionId: string;
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
      sessionId: session.sessionId,
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

  // ===== 服务状态监控（管理端）=====

  /** 查询 ASR/TTS 引擎服务状态（需 voice:service:monitor 权限，管理员） */
  static getServiceStatus() {
    return request<ServiceStatusVO>({
      url: "/api/v1/voice/service/status",
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

  // ===== 引擎注册表管理（管理端，voice:engine:manage）=====

  /** 引擎分页列表 */
  static listProviders(params: VoiceProviderPageQuery) {
    return request<PageResult<VoiceProviderVO[]>>({
      url: "/api/v1/voice/providers",
      method: "get",
      params,
    });
  }

  /** 指定能力维度的启用引擎列表 */
  static listEnabledProviders(engineType: string) {
    return request<VoiceProviderVO[]>({
      url: "/api/v1/voice/providers/enabled",
      method: "get",
      params: { engineType },
    });
  }

  /** 新增引擎（provider_code 唯一，删除后不可复用） */
  static createProvider(data: VoiceProviderCreateForm) {
    return request<VoiceProviderVO>({
      url: "/api/v1/voice/providers",
      method: "post",
      data,
    });
  }

  /** 更新引擎（provider_code 与 engine_type 不可变更） */
  static updateProvider(providerId: number, data: VoiceProviderUpdateForm) {
    return request<VoiceProviderVO>({
      url: `/api/v1/voice/providers/${providerId}`,
      method: "put",
      data,
    });
  }

  /** 删除引擎（存在启用模型引用时后端拒绝） */
  static deleteProvider(providerId: number) {
    return request({
      url: `/api/v1/voice/providers/${providerId}`,
      method: "delete",
    });
  }

  /** 引擎连通性测试（结果仅提示不阻断） */
  static testProviderConnection(providerId: number) {
    return request<VoiceProviderTestResultVO>({
      url: `/api/v1/voice/providers/${providerId}/test-connection`,
      method: "post",
    });
  }

  /** 查询引擎 API Key 列表 */
  static listProviderKeys(providerId: number) {
    return request<VoiceProviderKeyVO[]>({
      url: `/api/v1/voice/providers/${providerId}/keys`,
      method: "get",
    });
  }

  /** 新增引擎 API Key（key 明文，加密存储后不再返回） */
  static createProviderKey(providerId: number, data: VoiceProviderKeyCreateForm) {
    return request<VoiceProviderKeyVO>({
      url: `/api/v1/voice/providers/${providerId}/keys`,
      method: "post",
      data,
    });
  }

  /** 更新引擎 API Key（状态/优先级/权重/限额等） */
  static updateProviderKey(providerId: number, keyId: number, data: VoiceProviderKeyUpdateForm) {
    return request<VoiceProviderKeyVO>({
      url: `/api/v1/voice/providers/${providerId}/keys/${keyId}`,
      method: "put",
      data,
    });
  }

  /** 删除引擎 API Key（物理删除） */
  static deleteProviderKey(providerId: number, keyId: number) {
    return request({
      url: `/api/v1/voice/providers/${providerId}/keys/${keyId}`,
      method: "delete",
    });
  }

  /** 模型/音色列表（按 engine_type 筛选） */
  static listVoiceModels(params?: VoiceModelPageQuery) {
    return request<VoiceModelVO[]>({
      url: "/api/v1/voice/models",
      method: "get",
      params,
    });
  }

  /** 新增模型/音色（同一引擎下 model_id 唯一，删除后不可复用） */
  static createVoiceModel(data: VoiceModelCreateForm) {
    return request<VoiceModelVO>({
      url: "/api/v1/voice/models",
      method: "post",
      data,
    });
  }

  /** 更新模型/音色（model_id 不可变更） */
  static updateVoiceModel(modelId: number, data: VoiceModelUpdateForm) {
    return request<VoiceModelVO>({
      url: `/api/v1/voice/models/${modelId}`,
      method: "put",
      data,
    });
  }

  /** 删除模型/音色（model_id 保留） */
  static deleteVoiceModel(modelId: number) {
    return request({
      url: `/api/v1/voice/models/${modelId}`,
      method: "delete",
    });
  }
}

export default VoiceAPI;
