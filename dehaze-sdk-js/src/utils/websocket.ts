/**
 * WebSocket 客户端封装
 *
 * 基于原生 WebSocket API，补充：
 * - 自动重连（可配置最大重试次数与间隔）
 * - 心跳保活（对齐后端 SSE 15s 心跳节奏，WS 场景可选）
 * - 二进制/文本双通道收发（ASR 场景：上行 PCM 二进制 + 下行 JSON 文本）
 * - 优雅关闭（等待 onclose 回调）
 *
 * 不引入第三方依赖（如 reconnecting-websocket），原生 API + 简单封装即可覆盖 ASR 场景。
 * 若后续场景复杂化（多通道、消息队列、背压控制），再评估引入第三方库。
 */

/** WebSocket 事件回调集合 */
export interface WSHandlers {
  /** 收到文本消息时触发（JSON 字符串，由调用方解析） */
  onMessage?: (data: string) => void;
  /** 收到二进制消息时触发 */
  onBinaryMessage?: (data: ArrayBuffer | Blob) => void;
  /** 连接建立时触发 */
  onOpen?: () => void;
  /** 连接关闭时触发（code/closeReason 供调用方判断是否需重连） */
  onClose?: (code: number, reason: string) => void;
  /** 连接错误时触发 */
  onError?: (error: Event) => void;
  /** 重连尝试时触发（attempt 为第几次重试） */
  onReconnect?: (attempt: number) => void;
}

/** WebSocket 客户端配置 */
export interface WSClientConfig {
  /** WebSocket 连接地址 */
  url: string;
  /** 子协议（可选） */
  protocols?: string | string[];
  /** 事件回调 */
  handlers: WSHandlers;
  /** 是否启用自动重连，默认 true */
  autoReconnect?: boolean;
  /** 最大重连次数，默认 3 */
  maxReconnectAttempts?: number;
  /** 重连间隔（毫秒），默认 3000 */
  reconnectInterval?: number;
}

class WSClient {
  private ws: WebSocket | null = null;
  private config: WSClientConfig;
  private reconnectAttempts = 0;
  private manualClosed = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config: WSClientConfig) {
    this.config = {
      autoReconnect: true,
      maxReconnectAttempts: 3,
      reconnectInterval: 3000,
      ...config,
    };
  }

  /** 建立连接 */
  connect(): void {
    this.manualClosed = false;
    this.reconnectAttempts = 0;
    this.doConnect();
  }

  /** 发送文本消息 */
  send(data: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  /** 发送二进制数据（PCM 音频块等） */
  sendBinary(data: ArrayBuffer | ArrayBufferView | Blob): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  /** 主动关闭连接（不触发重连） */
  close(code?: number, reason?: string): void {
    this.manualClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close(code, reason);
  }

  /** 当前连接是否已建立 */
  isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private doConnect(): void {
    this.ws = new WebSocket(this.config.url, this.config.protocols);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.config.handlers.onOpen?.();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === "string") {
        this.config.handlers.onMessage?.(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        this.config.handlers.onBinaryMessage?.(event.data);
      } else if (event.data instanceof Blob) {
        this.config.handlers.onBinaryMessage?.(event.data);
      }
    };

    this.ws.onerror = (error: Event) => {
      this.config.handlers.onError?.(error);
    };

    this.ws.onclose = (event: CloseEvent) => {
      this.config.handlers.onClose?.(event.code, event.reason);
      if (!this.manualClosed && this.config.autoReconnect) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts!) {
      return;
    }
    this.reconnectAttempts++;
    this.config.handlers.onReconnect?.(this.reconnectAttempts);
    this.reconnectTimer = setTimeout(() => {
      this.doConnect();
    }, this.config.reconnectInterval);
  }
}

/**
 * 创建 WebSocket 客户端。
 *
 * 用于语音交互模块的流式 ASR（上行 PCM 二进制 + 下行 JSON 文本）等 WebSocket 场景。
 *
 * @example
 * ```ts
 * const ws = createWebSocket({
 *   url: "ws://host/ws/asr",
 *   handlers: {
 *     onMessage: (data) => {
 *       const result = JSON.parse(data); // { text, is_final }
 *     },
 *     onOpen: () => {
 *       ws.sendBinary(pcmAudioChunk);
 *     },
 *   },
 * });
 * ws.connect();
 * // 结束时：ws.send("EOS"); ws.close();
 * ```
 */
export function createWebSocket(config: WSClientConfig): WSClient {
  return new WSClient(config);
}

export { WSClient };
