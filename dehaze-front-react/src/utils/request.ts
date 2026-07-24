import {
  configJavaAxios,
  configPythonAxios,
  ResponseData,
  ResultEnum,
  TOKEN_KEY,
} from "dehaze-sdk-js";

import type { AxiosError, InternalAxiosRequestConfig } from "axios";

import store from "@/store";
import { resetToken } from "@/store/modules/userSlice";

import { message, Modal } from "antd";

const onResponseError = (error: AxiosError) => {
  if (error.response?.data) {
    const { code, msg } = error.response.data as ResponseData;
    if (code === ResultEnum.TOKEN_INVALID) {
      Modal.confirm({
        title: "提示",
        content: "登录已失效，请重新登录",
        onOk() {
          store.dispatch(resetToken());
          window.location.href = `/login?redirect=${encodeURIComponent(window.location.href)}`;
        },
        onCancel() {
          store.dispatch(resetToken());
        },
      });
    } else {
      message.error(msg || "系统出错");
    }
  }
  return Promise.reject(error.message);
};

export default function configRequest() {
  configJavaAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_JAVA_BASE_API,
      };
    },
    onResponseError,
    getToken: () => localStorage.getItem(TOKEN_KEY),
  });
  configPythonAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_PYTHON_BASE_API,
      };
    },
    onResponseError,
    getToken: () => localStorage.getItem(TOKEN_KEY),
  });
}
