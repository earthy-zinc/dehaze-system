import { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios";

export interface InterceptorCallbacks {
  onRequest?: (config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig;
  onRequestError?: (error: AxiosError) => AxiosError;

  onResponse?: (response: AxiosResponse) => AxiosResponse;
  onResponseError?: (error: AxiosError) => AxiosError;
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
