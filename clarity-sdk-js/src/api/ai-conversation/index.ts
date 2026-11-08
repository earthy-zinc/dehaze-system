import type { PageResult } from "../../types";
import request, { service } from "../../utils/request";
import {
  fetchSSE,
  isAIChatStreamEvent,
  type SSEEvent,
  type SSEStreamOptions,
} from "../../utils/sse";
import axios, { type InternalAxiosRequestConfig } from "axios";
import { configManager } from "../../config";
import { generateTraceId } from "../../logger";
import type {
  Artifact,
  BlockDeltaEvent,
  BlockStartEvent,
  BlockStopEvent,
  ClaudeMessageForm,
  CompatCallQuery,
  ConversationCreateForm,
  ConversationQuery,
  ConversationUpdateForm,
  ConversationVO,
  EditMessageForm,
  ErrorEvent,
  Feedback,
  FeedbackForm,
  Interrupt,
  InterruptEvent,
  MemoryCreateForm,
  MemoryQuery,
  MemoryUpdateForm,
  MemoryVO,
  Message,
  MessageEndEvent,
  MessageList,
  MessageListQuery,
  MessageResume,
  MessageSend,
  MessageStartEvent,
  OpenAICompletionForm,
  PlanUpdateEvent,
  SuggestionsEvent,
  UsageEvent,
} from "./model";

/** SSE 事件回调集合（回调名与线上事件名一一对应） */
export interface MessageStreamHandlers {
  /** message.start：本轮开始，携带用户消息与助手消息（均已落库） */
  onStart?: (data: MessageStartEvent) => void;
  /** block.start：块开始（text/thinking/tool_use，同一消息内按 seq 排序） */
  onBlockStart?: (data: BlockStartEvent) => void;
  /** block.delta：块增量，调用方按 blockId 自行累加文本 */
  onBlockDelta?: (data: BlockDeltaEvent) => void;
  /** block.stop：块结束，携带落库后的完整块（工具结果、耗时已就位） */
  onBlockStop?: (data: BlockStopEvent) => void;
  /** plan.update：计划全量推送（含修订记录） */
  onPlanUpdate?: (data: PlanUpdateEvent) => void;
  /** interrupt：推理中断（confirm/quota/async_wait/plan_approve），可查可续 */
  onInterrupt?: (data: InterruptEvent) => void;
  /** usage：本轮用量（可多次推送，最后一次为终值） */
  onUsage?: (data: UsageEvent) => void;
  /** suggestions：回复完成后的追问建议 */
  onSuggestions?: (data: SuggestionsEvent) => void;
  /** ping：心跳保活 */
  onPing?: () => void;
  /** error：业务错误事件（模型超载/内容过滤等），≠ 本轮终态，终态由 message.end 表达 */
  onError?: (data: ErrorEvent) => void;
  /** message.end：本轮终态（completed/failed/canceled/awaiting_input），每轮最多一次 */
  onEnd?: (data: MessageEndEvent) => void;
  /**
   * 每个 SSE 事件携带的递增 ID（后端 `id:` 帧）。
   *
   * 调用方需自行记录最后一次收到的 ID 并传给 `reconnectStream` 的 lastEventId，
   * 否则重连时服务端从 0 重放整条流。每个事件都会触发，早于同类事件回调。
   */
  onEventId?: (id: string) => void;
  /** 网络层错误（连接失败/断开/HTTP 非 2xx），流已中断 */
  onNetworkError?: (error: Error) => void;
  /** 流式正常结束 */
  onClose?: () => void;
}

/**
 * 创建单条流的事件分发器。
 *
 * 终态唯一：`message.end` 到达后本轮结束，后续帧（重放重叠、迟到心跳）一律丢弃，
 * 避免同一条消息被二次置为终态；唯一例外是 `suggestions` —— 推荐追问由后端在
 * message.end 之后异步生成（需调用 LLM），它只补充消息内容、不改变终态。
 * `error` 不是终态，仍可能后接 `message.end`。
 */
function createStreamDispatcher(handlers: MessageStreamHandlers): (event: SSEEvent) => void {
  let ended = false;
  return (event: SSEEvent) => {
    if (event.id !== undefined) handlers.onEventId?.(event.id);
    const name = event.event;
    // suggestions 是终态后的内容补全，不能因 ended 丢弃；其余事件一律丢弃。
    if (ended && name !== "suggestions") return;
    if (!event.data) return;
    if (!isAIChatStreamEvent(name)) return;
    let payload: unknown;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    switch (name) {
      case "message.start":
        handlers.onStart?.(payload as MessageStartEvent);
        break;
      case "block.start":
        handlers.onBlockStart?.(payload as BlockStartEvent);
        break;
      case "block.delta":
        handlers.onBlockDelta?.(payload as BlockDeltaEvent);
        break;
      case "block.stop":
        handlers.onBlockStop?.(payload as BlockStopEvent);
        break;
      case "plan.update":
        handlers.onPlanUpdate?.(payload as PlanUpdateEvent);
        break;
      case "interrupt":
        handlers.onInterrupt?.(payload as InterruptEvent);
        break;
      case "usage":
        handlers.onUsage?.(payload as UsageEvent);
        break;
      case "suggestions":
        handlers.onSuggestions?.(payload as SuggestionsEvent);
        break;
      case "message.end":
        ended = true;
        handlers.onEnd?.(payload as MessageEndEvent);
        break;
      case "error":
        handlers.onError?.(payload as ErrorEvent);
        break;
      case "ping":
        handlers.onPing?.();
        break;
    }
  };
}

/** 组装 fetchSSE 的接收入参：事件分发器 + 网络错误 + 关闭 */
function toSSEHandlers(handlers: MessageStreamHandlers) {
  return {
    onEvent: createStreamDispatcher(handlers),
    onError: (error: Error) => handlers.onNetworkError?.(error),
    onClose: () => handlers.onClose?.(),
  };
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
  static createConversation(data: ConversationCreateForm = {}) {
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

  /**
   * 会话详情（含模型配置、消息数等）
   *
   * query.view=admin 为管理端审计视角，读取任意用户会话并带审计字段（需
   * `ai:conversation:audit`），省略为当前用户。
   */
  static getConversation(id: number, query?: { view?: "admin" }) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${id}`,
      method: "get",
      params: query,
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

  /**
   * 批量操作会话（归档/恢复/删除）。
   *
   * action=delete 时需 `confirm=true` 二次确认，删除后进回收站。
   */
  static batchConversations(data: {
    action: "archive" | "restore" | "delete";
    ids: number[];
    confirm?: boolean;
  }) {
    return request<number>({
      url: "/api/v1/ai/conversations/batch",
      method: "post",
      data,
    });
  }

  /** 回收站会话列表（分页，含已删除会话） */
  static getTrashConversations(query?: ConversationQuery) {
    return request<PageResult<ConversationVO[]>>({
      url: "/api/v1/ai/conversations/trash",
      method: "get",
      params: query,
    });
  }

  /** 导出会话记录（format=markdown|json，默认 markdown；推理过程默认不导出），返回 Blob 下载 */
  static exportConversation(id: number, format: "json" | "markdown" = "markdown") {
    return request<Blob>({
      url: `/api/v1/ai/conversations/${id}/export`,
      method: "get",
      params: { format },
      responseType: "blob",
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
   * @param options 可选注入（fetchImpl，供测试注入假 fetch；缺省用全局 fetch）
   * @returns AbortController（用于中断流式）
   */
  static sendMessage(
    conversationId: number,
    data: MessageSend,
    handlers: MessageStreamHandlers,
    options?: SSEStreamOptions
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
        ...(options?.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      },
      toSSEHandlers(handlers)
    );

    return controller;
  }

  /**
   * 恢复中断的推理（SSE 续流）。
   *
   * 请求体即契约 `MessageResume`（confirm/params/planEdit 全 camelCase），原样上行。
   *
   * @param messageId 中断的消息 ID
   * @param data 恢复表单
   * @param handlers SSE 事件回调（同 sendMessage）
   * @param options 可选注入（fetchImpl，供测试注入假 fetch；缺省用全局 fetch）
   * @returns AbortController
   */
  static resumeMessage(
    messageId: number,
    data: MessageResume,
    handlers: MessageStreamHandlers,
    options?: SSEStreamOptions
  ): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}/resume`),
        method: "POST",
        body: data,
        headers: buildSSEHeaders(),
        signal: controller.signal,
        ...(options?.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      },
      toSSEHandlers(handlers)
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
   * @param options 可选注入（fetchImpl，供测试注入假 fetch；缺省用全局 fetch）
   * @returns 本次重连连接的 AbortController（调用方需替换持有的旧连接引用）
   */
  static reconnectStream(
    conversationId: number,
    streamSessionId: string,
    lastEventId: string,
    handlers: MessageStreamHandlers,
    options?: SSEStreamOptions
  ): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(
          `/api/v1/ai/conversations/${conversationId}/messages/stream/${streamSessionId}`
        ),
        method: "GET",
        headers: buildSSEHeaders(),
        lastEventId,
        signal: controller.signal,
        ...(options?.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      },
      toSSEHandlers(handlers)
    );
    return controller;
  }

  /**
   * 会话消息列表（游标分页，按 id 倒序返回）。
   *
   * 向上增量加载更早历史：以本页最小 id 作为下一次的 `before`。
   * 返回即契约 `MessageList`，历史消息是完整 `Message`（含 blocks/plan/pendingInterrupt/usage）。
   */
  static getMessages(conversationId: number, query?: MessageListQuery) {
    return request<MessageList>({
      url: `/api/v1/ai/conversations/${conversationId}/messages`,
      method: "get",
      params: query,
    });
  }

  /** 消息详情（完整 Message，含块、计划、待处理中断与用量） */
  static getMessageDetail(messageId: number, query?: { view?: "admin" }) {
    return request<Message>({
      url: `/api/v1/ai/messages/${messageId}`,
      method: "get",
      params: query,
    });
  }

  /**
   * 查询消息当前待处理的中断。
   *
   * 进入会话时补查（SSE 中断事件只在推流当下出现）；无待处理中断返回 null，
   * 超出 `expiresAt` 的中断 `actionable=false`，不可再 resume。
   */
  static getMessageInterrupt(messageId: number) {
    return request<Interrupt | null>({
      url: `/api/v1/ai/messages/${messageId}/interrupt`,
      method: "get",
    });
  }

  /** 某消息的分支列表（全部兄弟消息，后端按时间倒序） */
  static getBranches(conversationId: number, messageId: number) {
    return request<Message[]>({
      url: `/api/v1/ai/conversations/${conversationId}/messages/${messageId}/branches`,
      method: "get",
    });
  }

  /** 切换当前激活分支（更新会话 currentBranchMessageId，返回更新后的会话） */
  static switchBranch(conversationId: number, messageId: number) {
    return request<ConversationVO>({
      url: `/api/v1/ai/conversations/${conversationId}/branches/${messageId}`,
      method: "put",
    });
  }

  /**
   * 重新生成回复（创建分支消息，SSE 流式）。
   *
   * @param options 可选注入（fetchImpl，供测试注入假 fetch；缺省用全局 fetch）
   * @returns AbortController
   */
  static regenerate(
    messageId: number,
    handlers: MessageStreamHandlers,
    options?: SSEStreamOptions
  ): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}/regenerate`),
        method: "POST",
        headers: buildSSEHeaders(),
        signal: controller.signal,
        ...(options?.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      },
      toSSEHandlers(handlers)
    );
    return controller;
  }

  /**
   * 编辑用户消息并重新触发回复（SSE 流式）。
   *
   * @param options 可选注入（fetchImpl，供测试注入假 fetch；缺省用全局 fetch）
   * @returns AbortController
   */
  static editMessage(
    messageId: number,
    data: EditMessageForm,
    handlers: MessageStreamHandlers,
    options?: SSEStreamOptions
  ): AbortController {
    const controller = new AbortController();
    void fetchSSE(
      {
        url: resolveApiUrl(`/api/v1/ai/messages/${messageId}`),
        method: "PUT",
        body: data,
        headers: buildSSEHeaders(),
        signal: controller.signal,
        ...(options?.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      },
      toSSEHandlers(handlers)
    );
    return controller;
  }

  /** 删除助手回复消息（软删除） */
  static deleteMessage(messageId: number) {
    return request({
      url: `/api/v1/ai/messages/${messageId}`,
      method: "delete",
    });
  }

  /** 停止流式输出 / 取消当前推理（返回终态 Message） */
  static stopMessage(messageId: number) {
    return request<Message>({
      url: `/api/v1/ai/messages/${messageId}/stop`,
      method: "post",
    });
  }

  // ==================== 上下文：产物与记忆 ====================

  /** 会话中间产物列表（分页） */
  static getArtifacts(conversationId: number, query?: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<Artifact[]>>({
      url: `/api/v1/ai/conversations/${conversationId}/artifacts`,
      method: "get",
      params: query,
    });
  }

  /** 消息关联产物列表 */
  static getMessageArtifacts(messageId: number) {
    return request<Artifact[]>({
      url: `/api/v1/ai/messages/${messageId}/artifacts`,
      method: "get",
    });
  }

  /** 按业务引用反查产物列表 */
  static getArtifactsByRef(refType: string, refId: number) {
    return request<Artifact[]>({
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

  /** 取消归档记忆（`archived` 置 0 并重置衰减计时器，恢复对话注入；未归档 A0502） */
  static unarchiveMemory(id: number) {
    return request<MemoryVO>({
      url: `/api/v1/ai/memories/${id}/unarchive`,
      method: "post",
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

  // ==================== 第三方兼容 ====================

  /** 兼容调用审计查询（当前登录用户，create_time 倒序分页） */
  static getCompatCalls(query?: CompatCallQuery) {
    return request<PageResult<Record<string, unknown>[]>>({
      url: "/api/v1/ai/compat/calls",
      method: "get",
      params: query,
    });
  }

  // ==================== 消息反馈 ====================

  /**
   * 会话内当前用户的全部反馈（批量）。
   *
   * 返回契约 `Feedback` 数组；会话内无反馈时为空数组（不是 undefined/null），
   * 供进入会话时一次性还原各消息的点赞/点踩状态，避免逐条查询。
   */
  static getConversationFeedback(conversationId: number) {
    return request<Feedback[]>({
      url: `/api/v1/ai/conversations/${conversationId}/feedback`,
      method: "get",
    });
  }

  /** 提交/更新反馈（点赞/点踩） */
  static submitFeedback(messageId: number, data: FeedbackForm) {
    return request<Feedback>({
      url: `/api/v1/ai/messages/${messageId}/feedback`,
      method: "post",
      data,
    });
  }

  /** 查询消息反馈状态 */
  static getFeedback(messageId: number) {
    return request<Feedback | undefined>({
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
