import { configAxios, ResponseData } from "dehaze-sdk-js";

import type { AxiosError } from "axios";

import { message } from "antd";

function createOnResponseError() {
  return (error: AxiosError) => {
    if (error.response?.data) {
      const { msg } = error.response.data as ResponseData;
      message.error(msg || "系统出错");
    } else if (error.request) {
      message.error("网络异常，请检查网络连接");
    } else {
      message.error(error.message || "请求发送失败");
    }
    return Promise.reject(error);
  };
}

export default function configRequest() {
  configAxios({
    onResponseError: createOnResponseError(),
  });
}
