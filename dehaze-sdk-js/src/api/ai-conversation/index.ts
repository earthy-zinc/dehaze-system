import { PageResult } from "@/types";
import request, { service } from "@/utils/request";
import { fetchSSE, type SSEEvent } from "@/utils/sse";
import axios, { type InternalAxiosRequestConfig } from "axios";
import { configManager } from "@/config";
import { generateTraceId } from "@/logger";
import type {
  AiModelForm,
  AiModelQuery,
  AiModelVO,
  ArtifactVO,
  ClaudeMessageForm,
  ConversationCreateForm,
  ConversationQuery,
  ConversationUpdateForm,
  ConversationVO,
  EditMessageForm,
  ErrorEvent,
  FeedbackForm,
  FeedbackVO,
  MemoryCreateForm,
  MemoryQuery,
  MemoryUpdateForm,
  MemoryVO,
  MessageEndEvent,
  MessageResumeForm,
  MessageStartEvent,
  AiMessageVO,
  PlanEvent,
  SendMessageForm,
  SuggestionsEvent,
  ThoughtEvent,
  ContentBlockStartEvent,
  ContentBlockDeltaEvent,
  ContentBlockStopEvent,
  InterruptEvent,
  OpenAICompletionForm,
} from "./model";

/** SSE 事件回调集合 */
export interface MessageStreamHandlers {
  /** message.start：消息开始（后端推送 messageId/conversationId/model） */
  onStart?: (data: MessageStartEvent) => void;
  /** content_block.start：内容块开始（text/thinking/tool_use） */
  onContentBlockStart?: (data: ContentBlockStartEvent) => void;
  /**
   * content_block.delta：内容块增量（逐 token 推送）。
   *
   * 调用方需自行累加文本：`text += data.delta.text`；思考增量读 `data.delta.thinking`。
   */
  onContentBlockDelta?: (data: ContentBlockDeltaEvent) => void;
  /** content_block.stop：内容块结束 */
  onContentBlockStop?: (data: ContentBlockStopEvent) => void;
  /** thought：推理步骤完成（status: 1 成功 / 2 失败 / 3 跳过） */
  onThought?: (data: ThoughtEvent) => void;
  /** plan：Plan-and-Execute 计划推送 */
  onPlan?: (data: PlanEvent) => void;
  /** suggestions：回复完成后推荐追问 */
  onSuggestions?: (data: SuggestionsEvent) => void;
  /** interrupt：推理中断（confirm/quota/async_wait/plan_approve） */
  onInterrupt?: (data: InterruptEvent) => void;
  /** ping：心跳保活 */
  onPing?: () => void;
  /** error：业务错误事件（SSE error 帧，如模型超载/内容过滤） */
  onError?: (data: ErrorEvent) => void;
  /** message.end：消息结束（含 stopReason 和 usage） */
  onEnd?: (data: MessageEndEvent) => void;
  /** 网络层错误（连接失败/断开/HTTP 非 2xx），流已中断 */
  onNetworkError?: (error: Error) => void;
  /** 流式正常结束 */
  onClose?: () => void;
}

/** 将 SSE 事件分发到对应回调 */
function dispatchSSEEvent(event: SSEEvent, handlers: MessageStreamHandlers): void {
  if (!event.data) return;
  let payload: unknown;
  try {
    payload = JSON.parse(event.data);
  } catch {
    return;
  }
  switch (event.event) {
    case "message.start":
      handlers.onStart?.(payload as MessageStartEvent);
      break;
    case "content_block.start":
      handlers.onContentBlockStart?.(payload as ContentBlockStartEvent);
      break;
    case "content_block.delta":
      handlers.onContentBlockDelta?.(payload as ContentBlockDeltaEvent);
      break;
    case "content_block.stop":
      handlers.onContentBlockStop?.(payload as ContentBlockStopEvent);
      break;
    case "thought":
      handlers.onThought?.(payload as ThoughtEvent);
      break;
    case "plan":
      handlers.onPlan?.(payload as PlanEvent);
      break;
    case "suggestions":
      handlers.onSuggestions?.(payload as SuggestionsEvent);
      break;
    case "interrupt":
      handlers.onInterrupt?.(payload as InterruptEvent);
      break;
    case "ping":
      handlers.onPing?.();
      break;
    case "error":
      handlers.onError?.(payload as ErrorEvent);
      break;
    case "message.end":
      handlers.onEnd?.(payload as MessageEndEvent);
      break;
  }
}

/**
 * 兼容 API 流式客户端。
 *
 * `service` 的响应拦截器会对 JSON 响应做业务码解包，原生流（ReadableStream）会被
 * 误判为业务错误而中断。这里复用 `service` 的默认配置与 `configManager.onRequest`
 * 请求拦截（baseURL/凭证注入），但移除响应解包，以保留 stream:true 返回可读流的语义。
 */
const compatStreamDefaults: {
  baseURL?: string;
  timeout?: number;
  withCredentials?: boolean;
} = {};
if (service.defaults.baseURL !== undefined) compatStreamDefaults.baseURL = service.defaults.baseURL;
if (service.defaults.timeout !== undefined) compatStreamDefaults.timeout = service.defaults.timeout;
if (service.defaults.withCredentials !== undefined) {
  compatStreamDefaults.withCredentials = service.defaults.withCredentials;
}
const compatStreamClient = axios.create({ ...compatStreamDefaults });
compatStreamClient.interceptors.request.use((config) => {
  config.headers.set("X-Trace-Id", generateTraceId());
  return configManager.getInterceptors().onRequest?.(config) || config;
});

/**
 * 将相对路径解析为绝对 URL。
 *
 * `fetchSSE` 直接使用原生 `fetch`，无法读取 axios `service` 实例的 baseURL，
 * 传入相对路径（如 `/api/v1/ai/...`）在浏览器/Node 均会被拒绝（Failed to parse URL）。
 * 此处基于 `service.defaults.baseURL`（由调用方 configAxios/测试注入）拼装绝对地址，
 * 保证 SSE 流式请求与内部 API 走同一 baseURL。
 */
function resolveApiUrl(path: string): string {
  const base = service.defaults.baseURL || "";
  if (/^https?:\/\//.test(path)) return path;
  // 去掉 base 末尾的 '/'，避免出现 `//api` 双斜杠
  return `${base.replace(/\/$/, "")}${path}`;
}

/**
 * 复用请求拦截器注入会话/认证请求头到 SSE 请求。
 *
 * `fetchSSE` 走原生 `fetch`，不经过 axios `service` 的请求拦截器，因而无法获得
 * `configAxios(onRequest)` 注入的 `X-Session-Id` 等鉴权头。这里构造一个最小 axios
 * 配置调用同一 onRequest 拦截器，将产出的请求头合并进 SSE 请求，保证流式请求与
 * 内部 API 使用一致的会话凭证。
 */
function buildSSEHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const config = {
    headers: { "Content-Type": "application/json;charset=utf-8" },
  } as unknown as InternalAxiosRequestConfig;
  const modified = configManager.getInterceptors().onRequest?.(config) || config;
  const headers: Record<string, string> = { ...extra };
  const cfgHeaders = modified.headers as Record<string, unknown> | undefined;
  if (cfgHeaders && typeof cfgHeaders === "object") {
    for (const [k, v] of Object.entries(cfgHeaders)) {
      if (typeof v === "string") headers[k] = v;
    }
  }
  return headers;
}

/**
 * AI 对话 API
 *
 * 内部 API（`/api/v1/ai`），含会话管理、SSE 流式消息、推理中断恢复、
 * 上下文产物与记忆、模型管理、消息反馈。
 *
 * OpenAI / Claude 兼容 API 仅供第三方接入，本系统前端使用内部 API。
 */
class AiConversationAPI {
  // ==================== 会话管理 ====================

  /** 创建对话会话 */
  static createConversation(data?: ConversationCreateForm) {
    return request<ConversationVO>({
      url: "/api/v1/ai/conversations",
      method: "post",
      data,
    });
  }

  /** 会话列表（分页，支持搜索/状态筛选） */
  static getConversations(query?: ConversationQuery) {
    return request<PageResult<ConversationVO[]>>({
      url: "/api/v1/ai/conversations",
      method: "get",
      params: query,
    });
  }

  /** 会话详情（含模型配置、消息数等） */
  static getConversation(id: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}`,
      method: "get",
    });
  }

  /** 部分更新会话（标题/置顶/状态/模型配置/Agent） */
  static updateConversation(id: number, data: ConversationUpdateForm) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}`,
      method: "patch",
      data,
    });
  }

  /** 删除会话（软删除，进回收站） */
  static deleteConversation(id: number) {
    return request({
      url: `/api/v1/ai/conversations/${id}`,
      method: "delete",
    });
  }

  /** 恢复软删会话 */
  static restoreConversation(id: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}/restore`,
      method: "post",
    });
  }

  /** 置顶会话 */
  static pinConversation(id: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}/pin`,
      method: "put",
    });
  }

  /** 取消置顶 */
  static unpinConversation(id: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}/unpin`,
      method: "put",
    });
  }

  /** 标记会话已读 */
  static markConversationRead(id: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}/read`,
      method: "put",
    });
  }

  // ==================== 消息（SSE 流式） ====================

  /**
   * 发送消息（SSE 流式输出）。
   *
   * - 自动携带 `Idempotency-Key`（UUID）防重复发送
   * - 返回 AbortController，调用 `.abort()` 中断流式（等效于 stop 接口）
   *
   * @param conversationId 会话 ID
   * @param data 消息表单（content/model）
   * @param handlers SSE 事件回调
   * @returns AbortController（用于中断流式）
   */
  static sendMessage(
    conversationId: number,
    data: SendMessageForm,
    handlers: MessageStreamHandlers
  ): AbortController {
    const controller = new AbortController();
    const idempotencyKey =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`;

    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/conversations/${conversationId}/messages`),
        method: "POST",
        body: data,
        headers: buildSSEHeaders({ "Idempotency-Key": idempotencyKey }),
        signal: controller.signal,
      },
      {
        onEvent: (event) => dispatchSSEEvent(event, handlers),
        onError: (error) => handlers.onNetworkError?.(error),
        onClose: () => handlers.onClose?.(),
      }
    );

    return controller;
  }

  /**
   * 恢复中断的推理（SSE 续流）。
   *
   * @param messageId 中断的消息 ID
   * @param data 恢复表单（confirm/params/planEdit）
   * @param handlers SSE 事件回调（同 sendMessage）
   * @returns AbortController
   */
  static resumeMessage(
    messageId: number,
    data: MessageResumeForm,
    handlers: MessageStreamHandlers
  ): AbortController {
    const controller = new AbortController();
    // 后端 MessageResume 为纯 BaseModel，plan_edit 以 snake_case 传输（无 camelCase 别名）；
    // 其余字段 confirm/params 与 wire 同名，直接透传。
    const { planEdit, ...rest } = data;
    const body: Record<string, unknown> = { ...rest };
    if (planEdit !== undefined) {
      body.plan_edit = planEdit;
    }
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}/resume`),
        method: "POST",
        body,
        headers: buildSSEHeaders(),
        signal: controller.signal,
      },
      {
        onEvent: (event) => dispatchSSEEvent(event, handlers),
        onError: (error) => handlers.onNetworkError?.(error),
        onClose: () => handlers.onClose?.(),
      }
    );
    return controller;
  }

  /**
   * SSE 断线重连。
   *
   * 携带 `Last-Event-ID` 请求头从断点恢复，重放已缓存但未送达的事件。
   *
   * @param conversationId 会话 ID
   * @param streamSessionId 流式会话 ID
   * @param lastEventId 最后收到的事件 ID
   * @param handlers SSE 事件回调（同 sendMessage）
   */
  static reconnectStream(
    conversationId: number,
    streamSessionId: string,
    lastEventId: string,
    handlers: MessageStreamHandlers
  ): void {
    void fetchSSE(
      {
        url: resolveApiUrl(
          `/api/v1/ai/conversations/${conversationId}/messages/stream/${streamSessionId}`
        ),
        method: "GET",
        headers: buildSSEHeaders(),
        lastEventId,
      },
      {
        onEvent: (event) => dispatchSSEEvent(event, handlers),
        onError: (error) => handlers.onNetworkError?.(error),
        onClose: () => handlers.onClose?.(),
      }
    );
  }

  /** 会话消息列表（分页，按时间正序） */
  static getMessages(conversationId: number, query?: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<AiMessageVO[]>>({
      url: `/api/v1/ai/conversations/${conversationId}/messages`,
      method: "get",
      params: query,
    });
  }

  /** 消息详情（含推理步骤、工具调用） */
  static getMessageDetail(messageId: number) {
    return request<AiMessageVO>({
      url: `/api/v1/ai/messages/${messageId}`,
      method: "get",
    });
  }

  /**
   * 重新生成回复（创建分支消息，SSE 流式）。
   *
   * @returns AbortController
   */
  static regenerate(messageId: number, handlers: MessageStreamHandlers): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}/regenerate`),
        method: "POST",
        headers: buildSSEHeaders(),
        signal: controller.signal,
      },
      {
        onEvent: (event) => dispatchSSEEvent(event, handlers),
        onError: (error) => handlers.onNetworkError?.(error),
        onClose: () => handlers.onClose?.(),
      }
    );
    return controller;
  }

  /**
   * 编辑用户消息并重新触发回复（SSE 流式）。
   *
   * @returns AbortController
   */
  static editMessage(
    messageId: number,
    data: EditMessageForm,
    handlers: MessageStreamHandlers
  ): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}`),
        method: "PUT",
        body: data,
        headers: buildSSEHeaders(),
        signal: controller.signal,
      },
      {
        onEvent: (event) => dispatchSSEEvent(event, handlers),
        onError: (error) => handlers.onNetworkError?.(error),
        onClose: () => handlers.onClose?.(),
      }
    );
    return controller;
  }

  /** 停止流式输出 / 取消当前推理 */
  static stopMessage(messageId: number) {
    return request<AiMessageVO>({
      url: `/api/v1/ai/messages/${messageId}/stop`,
      method: "post",
    });
  }

  // ==================== 上下文：产物与记忆 ====================

  /** 会话中间产物列表（分页） */
  static getArtifacts(conversationId: number, query?: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<ArtifactVO[]>>({
      url: `/api/v1/ai/conversations/${conversationId}/artifacts`,
      method: "get",
      params: query,
    });
  }

  /** 消息关联产物列表 */
  static getMessageArtifacts(messageId: number) {
    return request<ArtifactVO[]>({
      url: `/api/v1/ai/messages/${messageId}/artifacts`,
      method: "get",
    });
  }

  /** 按业务引用反查产物列表 */
  static getArtifactsByRef(refType: string, refId: number) {
    return request<ArtifactVO[]>({
      url: "/api/v1/ai/artifacts/by-ref",
      method: "get",
      params: { refType, refId },
    });
  }

  /** 中间产物详情（含运行时图片 URL 等元数据） */
  static getArtifactDetail(id: number) {
    return request<Record<string, unknown>>({
      url: `/api/v1/ai/artifacts/${id}/detail`,
      method: "get",
    });
  }

  /** 当前用户长期记忆列表（分页） */
  static getMemories(query?: MemoryQuery) {
    return request<PageResult<MemoryVO[]>>({
      url: "/api/v1/ai/memories",
      method: "get",
      params: query,
    });
  }

  /** 归档记忆分页列表 */
  static getArchivedMemories(query?: MemoryQuery) {
    return request<PageResult<MemoryVO[]>>({
      url: "/api/v1/ai/memories/archived",
      method: "get",
      params: query,
    });
  }

  /** 创建记忆 */
  static createMemory(data: MemoryCreateForm) {
    return request<MemoryVO>({
      url: "/api/v1/ai/memories",
      method: "post",
      data,
    });
  }

  /** 更新记忆 */
  static updateMemory(id: number, data: MemoryUpdateForm) {
    return request<MemoryVO>({
      url: `/api/v1/ai/memories/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除单条记忆（软删除，不再注入对话） */
  static deleteMemory(id: number) {
    return request({
      url: `/api/v1/ai/memories/${id}`,
      method: "delete",
    });
  }

  /** 关键词搜索记忆 */
  static searchMemories(keyword: string, limit = 5) {
    return request<MemoryVO[]>({
      url: "/api/v1/ai/memories/search",
      method: "get",
      params: { keyword, limit },
    });
  }

  /** 批量清空记忆（confirm 二次确认；30 天内可恢复） */
  static clearMemories(
    data?: { memoryType?: string; start?: string; end?: string },
    confirm = false
  ) {
    return request<number>({
      url: "/api/v1/ai/memories/clear",
      method: "post",
      params: { ...data, confirm },
    });
  }

  /** 恢复软删记忆 */
  static restoreMemories(
    data?: { memoryType?: string; start?: string; end?: string },
    confirm = false
  ) {
    return request<number>({
      url: "/api/v1/ai/memories/restore",
      method: "post",
      params: { ...data, confirm },
    });
  }

  /** 导出全部记忆（JSON/Markdown），返回 Blob 下载 */
  static exportMemories(fmt: "json" | "markdown" = "json") {
    return request<Blob>({
      url: "/api/v1/ai/memories/export",
      method: "get",
      params: { fmt },
      responseType: "blob",
    });
  }

  // ==================== 模型管理 ====================

  /** 模型分页列表（管理端） */
  static getModels(query?: AiModelQuery) {
    return request<PageResult<AiModelVO[]>>({
      url: "/api/v1/ai/models",
      method: "get",
      params: query,
    });
  }

  /** 启用模型列表（用户端，含 VIP 过滤） */
  static getEnabledModels() {
    return request<AiModelVO[]>({
      url: "/api/v1/ai/models/enabled",
      method: "get",
    });
  }

  /** 新增模型配置（管理员） */
  static createModel(data: AiModelForm) {
    return request<AiModelVO>({
      url: "/api/v1/ai/models",
      method: "post",
      data,
    });
  }

  /** 更新模型配置（管理员；model_id 为业务标识） */
  static updateModel(modelId: string, data: Partial<AiModelForm>) {
    return request<AiModelVO>({
      url: `/api/v1/ai/models/${modelId}`,
      method: "put",
      data,
    });
  }

  /** 删除模型配置（管理员，软删除，model_id 不可复用） */
  static deleteModel(modelId: string) {
    return request({
      url: `/api/v1/ai/models/${modelId}`,
      method: "delete",
    });
  }

  // ==================== 消息反馈 ====================

  /** 提交/更新反馈（点赞/点踩） */
  static submitFeedback(messageId: number, data: FeedbackForm) {
    return request<FeedbackVO>({
      url: `/api/v1/ai/messages/${messageId}/feedback`,
      method: "post",
      data,
    });
  }

  /** 查询消息反馈状态 */
  static getFeedback(messageId: number) {
    return request<FeedbackVO | undefined>({
      url: `/api/v1/ai/messages/${messageId}/feedback`,
      method: "get",
    });
  }

  /** 撤销反馈 */
  static deleteFeedback(messageId: number) {
    return request({
      url: `/api/v1/ai/messages/${messageId}/feedback`,
      method: "delete",
    });
  }

  // ==================== OpenAI 兼容 API（第三方接入） ====================

  /**
   * OpenAI 兼容对话补全。
   *
   * 认证：`Authorization: Bearer dhak_xxx`（API Key），经 `apiKey` 参数传入；
   * 未传时走会话凭证。
   *
   * @param data 对话补全请求
   * @param apiKey 可选，API Key（Bearer 认证）
   * @returns 非流式返回 OpenAI 格式 JSON；流式返回可读流（需自行解析 OpenAI SSE 格式）
   */
  static async openaiCompletion(data: OpenAICompletionForm, apiKey?: string) {
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    if (data.stream) {
      const response = await compatStreamClient.request<unknown, { data: unknown }>({
        url: "/api/v1/chat/completions",
        method: "POST",
        data,
        responseType: "stream",
        headers,
      });
      return response.data;
    }
    return request<unknown>({
      url: "/api/v1/chat/completions",
      method: "post",
      data,
      headers,
    });
  }

  /** OpenAI 格式模型列表 */
  static openaiModels(apiKey?: string) {
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }
    return request<unknown>({
      url: "/api/v1/models",
      method: "get",
      headers,
    });
  }

  // ==================== Claude 兼容 API（第三方接入） ====================

  /**
   * Claude 兼容消息对话。
   *
   * 认证：`x-api-key: dhak_xxx` + `anthropic-version`，经 `apiKey` 参数传入；
   * 未传时走会话凭证。
   *
   * @param data 消息对话请求
   * @param apiKey 可选，API Key（x-api-key 认证）
   * @returns 非流式返回 Claude 格式 JSON；流式返回可读流
   */
  static async claudeMessage(data: ClaudeMessageForm, apiKey?: string) {
    const headers: Record<string, string> = { "anthropic-version": "2023-06-01" };
    if (apiKey) {
      headers["x-api-key"] = apiKey;
    }
    if (data.stream) {
      const response = await compatStreamClient.request<unknown, { data: unknown }>({
        url: "/api/v1/messages",
        method: "POST",
        data,
        responseType: "stream",
        headers,
      });
      return response.data;
    }
    return request<unknown>({
      url: "/api/v1/messages",
      method: "post",
      data,
      headers,
    });
  }
}

export default AiConversationAPI;
