// API 模块导入
import AiAgentAPI from "./src/api/ai-agent";
import AiBillingAPI from "./src/api/ai-billing";
import AiConversationAPI from "./src/api/ai-conversation";
import AiKnowledgeBaseAPI from "./src/api/ai-knowledge-base";
import AiProviderAPI from "./src/api/ai-provider";
import AiScheduleAPI from "./src/api/ai-schedule";
import AlgorithmAPI from "./src/api/algorithm";
import ApiKeyAPI from "./src/api/api-key";
import AuthAPI from "./src/api/auth";
import DatasetAPI, { DatasetItemAPI, ItemFileAPI } from "./src/api/dataset";
import DeptAPI from "./src/api/dept";
import DictAPI from "./src/api/dict";
import FavoriteAPI from "./src/api/favorite";
import FeedbackAPI from "./src/api/feedback";
import FileAPI from "./src/api/file";
import ImageInputHistoryAPI from "./src/api/image-input";
import ImportExportAPI from "./src/api/import-export";
import MemberAPI from "./src/api/member";
import MenuAPI from "./src/api/menu";
import MessageAPI, {
  AnnouncementAPI,
  MessageTemplateAPI,
  NotificationSettingAPI,
} from "./src/api/message";
import ModelAPI from "./src/api/model";
import OrderAPI from "./src/api/order";
import PackageAPI, { CouponAPI, PromotionAPI } from "./src/api/package";
import RecommendationAPI from "./src/api/recommendation";
import RoleAPI from "./src/api/role";
import TaskAPI from "./src/api/task";
import UserAPI from "./src/api/user";
import VoiceAPI from "./src/api/voice";
export type { AsrStreamSession } from "./src/api/voice";

// API 模型导出
export * from "./src/api/ai-agent/model";
export * from "./src/api/ai-billing/model";
export * from "./src/api/ai-conversation/model";
export * from "./src/api/ai-knowledge-base/model";
export * from "./src/api/ai-provider/model";
export * from "./src/api/ai-schedule/model";
export * from "./src/api/algorithm/model";
export * from "./src/api/api-key/model";
export * from "./src/api/auth/model";
export * from "./src/api/dataset/model";
export * from "./src/api/dept/model";
export * from "./src/api/dict/model";
export * from "./src/api/favorite/model";
export * from "./src/api/feedback/model";
export * from "./src/api/file/model";
export * from "./src/api/image-input/model";
export * from "./src/api/import-export/model";
export * from "./src/api/member/model";
export * from "./src/api/menu/model";
export * from "./src/api/message/model";
export * from "./src/api/model/model";
export * from "./src/api/order/model";
export * from "./src/api/package/model";
export * from "./src/api/recommendation/model";
export * from "./src/api/role/model";
export * from "./src/api/task/model";
export * from "./src/api/user/model";
export * from "./src/api/voice/model";
export * from "./src/types";
export * from "./src/enums";

// 前端日志监控：Logger 多 transport 架构 + 错误捕获 + 离线上报
export { Logger, ErrorBoundary, ConsoleTransport, RemoteTransport } from "./src/logger";
export type { LogEntry, LogLevel, LoggerStorage, LogTransport, InstallConfig } from "./src/logger";

// 配置导出
export { configAxios } from "./src/config";

// Axios 实例导出
export { service } from "./src/utils/request";

// SSE 流式工具导出（供 AI 对话 SSE 流式消息使用）
export { fetchSSE } from "./src/utils/sse";
export type { SSEEvent, SSERequestConfig, SSEHandlers } from "./src/utils/sse";
export type { MessageStreamHandlers } from "./src/api/ai-conversation";

// WebSocket 工具导出（供语音交互流式 ASR 等场景使用）
export { createWebSocket } from "./src/utils/websocket";
export type { WSClient, WSClientConfig, WSHandlers } from "./src/utils/websocket";

// Axios 类型导出（供宿主项目配置 adapter / 拦截器使用）
export type { AxiosAdapter, AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios";

// API 导出
export {
  AiAgentAPI,
  AiBillingAPI,
  AiConversationAPI,
  AiKnowledgeBaseAPI,
  AiProviderAPI,
  AiScheduleAPI,
  AlgorithmAPI,
  ApiKeyAPI,
  AnnouncementAPI,
  AuthAPI,
  CouponAPI,
  DatasetAPI,
  DatasetItemAPI,
  ItemFileAPI,
  DeptAPI,
  DictAPI,
  FavoriteAPI,
  FeedbackAPI,
  FileAPI,
  ImageInputHistoryAPI,
  ImportExportAPI,
  MemberAPI,
  MenuAPI,
  MessageAPI,
  MessageTemplateAPI,
  ModelAPI,
  NotificationSettingAPI,
  OrderAPI,
  PackageAPI,
  PromotionAPI,
  RecommendationAPI,
  RoleAPI,
  TaskAPI,
  UserAPI,
  VoiceAPI,
};
