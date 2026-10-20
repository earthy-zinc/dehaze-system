/**
 * API 配置
 *
 * H5 与非 H5 统一使用完整后端地址直连（后端已配置 CORS）。
 * H5 开发环境的 vite proxy 仅作为 `/api` 兜底代理。
 */

/** 后端服务地址（来自 .env 的 VITE_API_HOST，未配置时回退默认值） */
export const API_HOST =
  import.meta.env.VITE_API_HOST || "http://127.0.0.1:8989";

/** API 版本前缀 */
const API_PREFIX = "/api/v1";

/** 用户信息本地存储 key */
export const USER_INFO_KEY = "user_info";

/** 会话失效事件：接口返回登录失效时广播，通知 auth store 清空内存态 */
export const SESSION_INVALID_EVENT = "auth:session-invalid";

/** 业务完整 baseURL（含 `/api/v1` 前缀） */
export const BASE_URL = `${API_HOST}${API_PREFIX}`;
