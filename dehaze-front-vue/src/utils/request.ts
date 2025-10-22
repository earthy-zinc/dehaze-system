import {
  configJavaAxios,
  configPythonAxios,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";

import type { AxiosError, InternalAxiosRequestConfig } from "axios";
import { useUserStoreHook } from "@/store/modules/user";

const onResponseError = (error: AxiosError) => {
  if (error.response?.data) {
    const { code, msg } = error.response.data as ResponseData;
    if (code === ResultEnum.TOKEN_INVALID) {
      ElMessageBox.confirm("当前页面已失效，请重新登录", "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      }).then(() => {
        const userStore = useUserStoreHook();
        userStore.resetToken().then(() => {
          location.reload();
        });
      });
    } else {
      ElMessage.error(msg || "系统出错");
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
  });
  configPythonAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_PYTHON_BASE_API,
      };
    },
    onResponseError,
  });
}
