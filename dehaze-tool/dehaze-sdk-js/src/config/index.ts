import { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios";

export interface InterceptorCallbacks {
  onRequest?: (config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig;
  onRequestError?: (error: AxiosError) => AxiosError;

  onResponse?: (response: AxiosResponse) => AxiosResponse | Promise<AxiosResponse>;
  onResponseError?: (error: AxiosError) => any;
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

export const configAxios = (callback: InterceptorCallbacks) => {
  configManager.setInterceptors(callback);
};
