import { vi } from "vitest";

/**
 * WebSocket Mock 类
 * 用于模拟 WebSocket 连接和消息推送
 */
export class MockWebSocket {
  public url: string;
  public readyState: number;
  public onopen: ((event: Event) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;

  public static CONNECTING = 0;
  public static OPEN = 1;
  public static CLOSING = 2;
  public static CLOSED = 3;

  private messageQueue: any[] = [];
  private closeCallback: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    this.readyState = MockWebSocket.CONNECTING;

    // 模拟异步连接
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event("open"));
      }
    }, 10);
  }

  /**
   * 发送消息
   */
  send(data: string | ArrayBuffer | Blob): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error("WebSocket is not open");
    }
    // 可以在这里添加发送消息的逻辑
  }

  /**
   * 关闭连接
   */
  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSING;
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose(new CloseEvent("close", { code, reason }));
      }
      if (this.closeCallback) {
        this.closeCallback();
      }
    }, 10);
  }

  /**
   * 模拟接收消息
   */
  simulateMessage(data: any): void {
    if (this.readyState === MockWebSocket.OPEN && this.onmessage) {
      const message = typeof data === "string" ? data : JSON.stringify(data);
      this.onmessage(
        new MessageEvent("message", {
          data: message,
        })
      );
    }
  }

  /**
   * 模拟错误
   */
  simulateError(error?: string): void {
    if (this.onerror) {
      this.onerror(new Event("error"));
    }
  }

  /**
   * 添加到消息队列
   */
  queueMessage(data: any): void {
    this.messageQueue.push(data);
  }

  /**
   * 发送队列中的所有消息
   */
  flushMessages(): void {
    this.messageQueue.forEach((data) => this.simulateMessage(data));
    this.messageQueue = [];
  }

  /**
   * 设置关闭回调
   */
  onCloseCallback(callback: () => void): void {
    this.closeCallback = callback;
  }
}

/**
 * 创建 WebSocket Mock 实例
 */
export function createMockWebSocket(): typeof MockWebSocket {
  return MockWebSocket as any;
}

/**
 * 模拟去雾任务的 WebSocket 进度消息
 */
export function createProgressMessage(
  taskId: string,
  progress: number,
  status: "processing" | "completed" | "failed" = "processing"
) {
  return {
    type: "progress",
    taskId,
    progress,
    status,
    timestamp: new Date().toISOString(),
  };
}

/**
 * 模拟去雾任务完成消息
 */
export function createCompletionMessage(
  taskId: string,
  resultUrl: string,
  metrics?: {
    psnr?: number;
    ssim?: number;
    processingTime?: number;
  }
) {
  return {
    type: "completed",
    taskId,
    status: "completed",
    progress: 100,
    resultImageUrl: resultUrl,
    metrics,
    timestamp: new Date().toISOString(),
  };
}

/**
 * 模拟去雾任务失败消息
 */
export function createErrorMessage(taskId: string, errorMessage: string) {
  return {
    type: "error",
    taskId,
    status: "failed",
    error: errorMessage,
    timestamp: new Date().toISOString(),
  };
}

/**
 * 设置全局 WebSocket Mock
 */
export function setupWebSocketMock(): {
  WebSocket: typeof MockWebSocket;
  getInstance: () => MockWebSocket | null;
} {
  let instance: MockWebSocket | null = null;

  const MockWebSocketConstructor = vi.fn((url: string) => {
    instance = new MockWebSocket(url);
    return instance;
  }) as any;

  // 复制静态属性
  MockWebSocketConstructor.CONNECTING = MockWebSocket.CONNECTING;
  MockWebSocketConstructor.OPEN = MockWebSocket.OPEN;
  MockWebSocketConstructor.CLOSING = MockWebSocket.CLOSING;
  MockWebSocketConstructor.CLOSED = MockWebSocket.CLOSED;

  global.WebSocket = MockWebSocketConstructor;

  return {
    WebSocket: MockWebSocketConstructor,
    getInstance: () => instance,
  };
}

/**
 * 模拟完整的去雾任务流程
 */
export async function simulateDehazeProgress(
  ws: MockWebSocket,
  taskId: string,
  steps: number = 5
): Promise<void> {
  return new Promise((resolve) => {
    let currentProgress = 0;
    const progressStep = 100 / steps;

    const interval = setInterval(() => {
      currentProgress += progressStep;

      if (currentProgress >= 100) {
        currentProgress = 100;
        ws.simulateMessage(
          createCompletionMessage(
            taskId,
            "https://via.placeholder.com/800x600",
            {
              psnr: 28.5,
              ssim: 0.92,
              processingTime: 3500,
            }
          )
        );
        clearInterval(interval);
        resolve();
      } else {
        ws.simulateMessage(createProgressMessage(taskId, currentProgress));
      }
    }, 100);
  });
}
