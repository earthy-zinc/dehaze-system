import { resetToken } from "@/store/user";
import { Dialog, Notify } from "@taroify/core";
import Taro from "@tarojs/taro";
import type { AxiosError } from "axios";
import {
  configJavaAxios,
  configPythonAxios,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";

const onResponseError = (error: AxiosError) => {
  if (error.response?.data) {
    const { code, msg } = error.response.data as ResponseData;
    if (code === ResultEnum.TOKEN_INVALID) {
      Dialog.confirm({
        title: "提示",
        message: "当前页面已失效，请重新登录",
        confirm: "确定",
        cancel: "取消",
        onConfirm: async () => {
          await resetToken();
          Taro.redirectTo({ url: "/pages/login/login" });
        },
      });
    } else {
      Notify.open(msg || "系统出错啦");
    }
  }
  return Promise.reject(error.message);
};

export default function configRequest() {
  configJavaAxios({
    onResponseError,
  });
  configPythonAxios({
    onResponseError,
  });
}
