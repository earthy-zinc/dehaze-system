/**
 * 前端日志字段规范（对齐 dehaze-doc/docs/05-改造计划/前端日志监控改造计划.md §3.3）
 */

export type LogLevel = "ERROR" | "WARN" | "INFO";

export interface LogEntry {
  /** ISO8601 UTC 时间戳 */
  timestamp: string;
  /** ERROR / WARN / INFO */
  level: LogLevel;
  /** 人读描述，仅放业务语义 */
  message: string;
  /** 固定 client，filebeat 注入 */
  service: "client";
  /** 前端项目标识：react/vue/taro/uniapp/rn/flutter/android */
  app: string;
  /** 应用版本号（未注入则不上报该字段） */
  app_version?: string;
  /** 当前页面 URL / 路由路径 */
  url: string;
  /** 浏览器/设备 User-Agent */
  user_agent: string;
  /** 与后端日志串联 */
  trace_id: string;
  /** 日志类型：performance 标识性能指标日志（默认普通错误/业务日志无此字段） */
  type?: "performance";
  /** 错误类型：js/promise/api */
  error_type?: "js" | "promise" | "api" | "resource";
  /** 错误来源 */
  error_source?: string;
  /** 完整堆栈字符串 */
  error_stack?: string;
  /** HTTP 方法（API 失败日志） */
  method?: string;
  /** 请求路径（不含 query） */
  path?: string;
  /** HTTP 状态码 */
  status?: number;
  /** 请求耗时（毫秒） */
  duration?: number;
  /** 业务错误码 */
  code?: string;
  /** 性能指标名（type=performance 日志，见改造计划 §3.3.5） */
  metric_name?: string;
  /** 性能指标值（毫秒；cls 无量纲，resource_error 固定 0） */
  metric_value?: number;
  /** 导航类型：navigate/reload/back_forward（仅页面加载类指标） */
  navigation_type?: string;
  /** 资源 URL（仅 resource_error / 资源加载耗时） */
  resource_url?: string;
}

export interface LoggerStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface LogTransport {
  /** 逐条立即输出（如 Console），不受采样/限流影响 */
  log?(entry: LogEntry): void;
  /** 批量上报（如 Remote），由 Logger 统一采样/限流后调用 */
  send?(logs: LogEntry[]): Promise<void>;
}

export interface InstallConfig {
  /** 前端项目标识：react/vue/taro/uniapp/rn */
  app: string;
  /** 应用版本号（构建时注入；不传则日志不含该字段） */
  appVersion?: string;
  /** transport 列表（应用端按需组装，缺省仅 ConsoleTransport） */
  transports?: LogTransport[];
  /** 自定义存储（缺省用 localStorage，离线缓存 key=dehaze_logs） */
  storage?: LoggerStorage;
  /** React 引用（ErrorBoundary 依赖，React 宿主项目注入） */
  react?: unknown;
  /** 单设备限流：窗口内最大上报条数（默认 20） */
  rateLimitMax?: number;
  /** 单设备限流窗口（毫秒，默认 60000） */
  rateLimitWindowMs?: number;
}
