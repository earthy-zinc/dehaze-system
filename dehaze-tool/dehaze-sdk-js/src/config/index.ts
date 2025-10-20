import { AxiosResponse, InternalAxiosRequestConfig } from "axios";

export interface InterceptorCallbacks {
  onRequest?: (
    config: InternalAxiosRequestConfig
  ) => InternalAxiosRequestConfig;
  onRequestError?: (error: any) => any;

  onResponse?: (response: AxiosResponse["data"]) => any;
  onResponseError?: (error: any) => any;
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
