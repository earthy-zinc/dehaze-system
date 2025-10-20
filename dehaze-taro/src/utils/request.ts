import { configJavaAxios, configPythonAxios } from "dehaze-sdk-js";
import { ResultEnum } from "../enums/ResultEnum";

export default function configRequest() {
  configJavaAxios({
    onResponse: (response) => {
      const { code, data, msg } = response;
      if (code !== ResultEnum.SUCCESS) {
        // ElMessage.error(msg || "系统出错");
        return Promise.reject(msg);
      }
      return data;
    },
    onResponseError: (error) => {
      if (error.response.data) {
        const { code, msg } = error.response.data;
        if (code === ResultEnum.TOKEN_INVALID) {
          console.log(msg);
          // ElMessageBox.confirm("当前页面已失效，请重新登录", "提示", {
          //   confirmButtonText: "确定",
          //   cancelButtonText: "取消",
          //   type: "warning",
          // }).then(() => {
          //   const userStore = useUserStoreHook();
          //   userStore.resetToken().then(() => {
          //     location.reload();
          //   });
          // });
        } else {
          // ElMessage.error(msg || "系统出错");
        }
      }
      return Promise.reject(error.message);
    },
  });
  configPythonAxios({});
}
