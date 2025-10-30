import {AxiosError, AxiosResponse, InternalAxiosRequestConfig} from "axios";

export interface InterceptorCallbacks {
    onRequest?: (
        config: InternalAxiosRequestConfig
    ) => InternalAxiosRequestConfig;
    onRequestError?: (error: AxiosError) => any;

    onResponse?: (response: AxiosResponse) => any;
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
export const pythonConfigManager = new ConfigManager();

export const configJavaAxios = (callback: InterceptorCallbacks) => {
    configManager.setInterceptors(callback);
};

export const configPythonAxios = (callback: InterceptorCallbacks) => {
    pythonConfigManager.setInterceptors(callback);
};
