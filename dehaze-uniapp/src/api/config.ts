/**
 * API 配置
 *
 * 不同平台的 baseURL 处理：
 * - H5 开发环境：使用 vite proxy，请求路径 `/api/v1` 会被代理到后端
 * - 小程序/App：需要完整 URL
 */

/** 后端服务地址（用于非 H5 平台的直连） */
const API_HOST = "http://127.0.0.1:8989";

/** 数据集静态文件服务地址（nginx-dataset，端口9000） */
const DATASET_HOST = "http://127.0.0.1:9000";

/** API 版本前缀 */
const API_PREFIX = "/api/v1";

/**
 * 获取当前平台的 API baseURL
 */
function getBaseURL(): string {
  // #ifdef H5
  // H5 开发环境使用 vite proxy，生产环境使用同源或 nginx 代理
  return API_PREFIX;
  // #endif

  // #ifndef H5
  // 小程序 / App 需要完整 URL
  return `${API_HOST}${API_PREFIX}`;
  // #endif
}

/** API 基础路径 */
export const BASE_URL = getBaseURL();

/**
 * 获取数据集静态文件服务 baseURL
 * 用于访问数据集图片（hazy/clean 等）
 */
function getDatasetBaseURL(): string {
  // #ifdef H5
  return "/dataset";
  // #endif

  // #ifndef H5
  return DATASET_HOST;
  // #endif
}

/** 数据集静态文件基础路径 */
export const DATASET_BASE_URL = getDatasetBaseURL();

/** 请求超时时间（毫秒） */
export const REQUEST_TIMEOUT = 30000;

/** 用户信息本地存储 key */
export const USER_INFO_KEY = "user_info";
