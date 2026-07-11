import { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios";

export interface InterceptorCallbacks {
  onRequest?: (config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig;
  onRequestError?: (error: AxiosError) => AxiosError;

  onResponse?: (response: AxiosResponse) => AxiosResponse | Promise<AxiosResponse>;
  /**
   * 响应错误处理。可返回一个 Promise 以恢复请求（如 token 刷新后重发），
   * 返回 undefined 则按原错误拒绝。
   */
  onResponseError?: (error: AxiosError) => any;

  /**
   * 获取 token（同步）。宿主端可注入自定义读取逻辑（如 Taro 同步存储）。
   * 未提供时回退到 localStorage.getItem(TOKEN_KEY)。
   */
  getToken?: () => string | null;
}

class ConfigManager {
  private callbacks: InterceptorCallbacks = {};

  setInterceptors(callbacks: InterceptorCallbacks) {
    this.callbacks = callbacks;
  }

  getInterceptors() {
    return this.callbacks;
  }
}

export const configManager = new ConfigManager();
export const pythonConfigManager = new ConfigManager();

export const configJavaAxios = (callback: InterceptorCallbacks) => {
  configManager.setInterceptors(callback);
};

export const configPythonAxios = (callback: InterceptorCallbacks) => {
  pythonConfigManager.setInterceptors(callback);
};
