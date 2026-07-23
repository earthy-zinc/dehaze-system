/**
 * SDK 配置
 *
 * 通过 dehaze-sdk-js 的 configJavaAxios / configPythonAxios 注入：
 * - baseURL（Java 主后端 / Python 辅助后端）
 * - token（同步读取，由 tokenStore 维护内存副本）
 * - 响应错误处理（token 失效触发回调）
 *
 * 响应拦截器约定：SDK 内部已校验 code===SUCCESS，并解包返回 response.data.data（业务载荷）。
 * 因此 onResponse 必须返回 response.data.data，与 LoginResult/CaptchaResult 等类型定义对齐。
 */
import { configJavaAxios, configPythonAxios, ResultEnum } from 'dehaze-sdk-js';
import { API_CONFIG } from './env';
import { tokenStore, triggerTokenInvalid } from '../utils/tokenStore';

/** SDK 拦截器配置类型（从 configJavaAxios 参数推导，避免依赖 SDK 内部类型导出） */
type InterceptorCallbacks = Parameters<typeof configJavaAxios>[0];

// token 失效码（统一使用 SDK ResultEnum，避免硬编码）
const TOKEN_INVALID_CODES: readonly string[] = [
  ResultEnum.TOKEN_INVALID,
  ResultEnum.ACCESS_UNAUTHORIZED,
  ResultEnum.FORBIDDEN_OPERATION,
];

function isTokenInvalid(error: unknown): boolean {
  const err = error as { response?: { status?: number; data?: { code?: string } } };
  const status = err.response?.status;
  const code = err.response?.data?.code;
  return status === 401 || (typeof code === 'string' && TOKEN_INVALID_CODES.includes(code));
}

/** 构造 SDK 拦截器配置（Java/Python 共用同一套逻辑，仅 baseURL 不同） */
function buildAxiosConfig(baseURL: string): InterceptorCallbacks {
  return {
    onRequest: config => {
      config.baseURL = baseURL;
      return config;
    },
    onRequestError: error => error,
    onResponse: response => response.data.data,
    onResponseError: error => {
      if (isTokenInvalid(error)) {
        tokenStore.clear();
        triggerTokenInvalid();
      }
      return Promise.reject(error);
    },
    getToken: () => tokenStore.get(),
  };
}

// 配置 Java 主后端
configJavaAxios(buildAxiosConfig(API_CONFIG.JAVA_BASE_URL));

// 配置 Python 辅助后端（算法推荐/收藏/对比）
configPythonAxios(buildAxiosConfig(API_CONFIG.PYTHON_BASE_URL));
