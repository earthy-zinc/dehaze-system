import { configJavaAxios, configPythonAxios, ResultEnum } from 'dehaze-sdk-js';
import { API_CONFIG } from './env';
import { sessionStore, triggerSessionInvalid } from '../utils/tokenStore';

type InterceptorCallbacks = Parameters<typeof configJavaAxios>[0];

const SESSION_INVALID_CODES: readonly string[] = [
  ResultEnum.TOKEN_INVALID,
  ResultEnum.ACCESS_UNAUTHORIZED,
  ResultEnum.FORBIDDEN_OPERATION,
];

function isSessionInvalid(error: unknown): boolean {
  const err = error as { response?: { status?: number; data?: { code?: string } } };
  const status = err.response?.status;
  const code = err.response?.data?.code;
  return status === 401 || (typeof code === 'string' && SESSION_INVALID_CODES.includes(code));
}

function buildAxiosConfig(baseURL: string): InterceptorCallbacks {
  return {
    onRequest: config => {
      config.baseURL = baseURL;
      const sid = sessionStore.get();
      if (sid) {
        config.headers.set('X-Session-Id', sid);
      }
      return config;
    },
    onRequestError: error => error,
    onResponse: response => response.data.data,
    onResponseError: error => {
      if (isSessionInvalid(error)) {
        sessionStore.clear();
        triggerSessionInvalid();
      }
      return Promise.reject(error);
    },
  };
}

configJavaAxios(buildAxiosConfig(API_CONFIG.JAVA_BASE_URL));

configPythonAxios(buildAxiosConfig(API_CONFIG.PYTHON_BASE_URL));
