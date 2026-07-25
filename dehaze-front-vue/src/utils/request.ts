import { configAxios, ResponseData } from "dehaze-sdk-js";

function createOnResponseError() {
  return (error: any) => {
    if (error.response?.data) {
      const { msg } = error.response.data as ResponseData;
      ElMessage.error(msg || "系统出错");
    } else if (error.request) {
      ElMessage.error("网络异常，请检查网络连接");
    } else {
      ElMessage.error(error.message || "请求发送失败");
    }
    return Promise.reject(error);
  };
}

export default function configRequest() {
  configAxios({
    onResponseError: createOnResponseError(),
  });
}
