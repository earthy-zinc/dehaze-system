import { configJavaAxios, configPythonAxios } from "dehaze-sdk-js";

import type { AxiosError } from "axios";
import { Dialog, Notify } from "@taroify/core";
import { ResultEnum } from "@/enums/ResultEnum";
export default function configRequest() {
  configJavaAxios({
    onResponseError: (error: AxiosError) => {
      if (error.response?.data) {
        const { code, msg } = error.response.data as any;
        if (code === ResultEnum.TOKEN_INVALID) {
          Dialog.confirm({
            title: "提示",
            message: "当前页面已失效，请重新登录",
            confirm: "确定",
            cancel: "取消",
            onConfirm: () => {
              // const userStore = useUserStoreHook();
              // userStore.resetToken().then(() => {
              //   location.reload();
              // });
            },
          });
        } else {
          Notify.open(msg || "系统出错啦");
        }
      }
      return Promise.reject(error.message);
    },
  });
  configPythonAxios({});
}
