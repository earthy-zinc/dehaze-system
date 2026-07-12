/**
 * SDK 配置
 *
 * 通过 dehaze-sdk-js 的 configJavaAxios / configPythonAxios 注入：
 * - baseURL（Java 主后端 / Python 辅助后端）
 * - token（同步读取，由 tokenStore 维护内存副本）
 * - 响应错误处理（token 失效触发回调）
 */
import { configJavaAxios, configPythonAxios } from 'dehaze-sdk-js';
import { API_CONFIG } from './env';
import { tokenStore, triggerTokenInvalid } from '../utils/tokenStore';

// token 失效码
const TOKEN_INVALID_CODES = ['A0230', 'A0301', 'A0302'];

function isTokenInvalid(error: any): boolean {
  const status = error.response?.status;
  const code = error.response?.data?.code;
  return status === 401 || (typeof code === 'string' && TOKEN_INVALID_CODES.includes(code));
}

// 配置 Java 主后端
configJavaAxios({
  onRequest: config => {
    config.baseURL = API_CONFIG.JAVA_BASE_URL;
    return config;
  },
  onRequestError: error => error,
  onResponse: response => response.data,
  onResponseError: error => {
    if (isTokenInvalid(error)) {
      tokenStore.clear();
      triggerTokenInvalid();
    }
    return Promise.reject(error);
  },
  getToken: () => tokenStore.get(),
});

// 配置 Python 辅助后端（算法推荐/收藏/对比）
configPythonAxios({
  onRequest: config => {
    config.baseURL = API_CONFIG.PYTHON_BASE_URL;
    return config;
  },
  onRequestError: error => error,
  onResponse: response => response.data,
  onResponseError: error => {
    if (isTokenInvalid(error)) {
      tokenStore.clear();
      triggerTokenInvalid();
    }
    return Promise.reject(error);
  },
  getToken: () => tokenStore.get(),
});
