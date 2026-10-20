import router from "@/router";
import { useUserStore } from "@/store/modules/user";
import { configAxios, ResponseData, ResultEnum } from "clarity-sdk-js";

/** 会话失效的返回码：token 无效/过期、token 被禁止访问 */
const SESSION_INVALID_CODES: string[] = [
  ResultEnum.TOKEN_INVALID,
  ResultEnum.TOKEN_ACCESS_FORBIDDEN,
];

// 失效风暴下并发请求会同时命中，只处理一次
let isRedirecting = false;

/**
 * 会话失效兜底：会话 cookie 是 HttpOnly（JS 不可清），`/auth/logout` 又要求有效会话，
 * 因此前端只能清内存用户态并把用户送回登录页（登录页守卫据 roles 已清空放行）。
 *
 * @returns 是否已接管本次失效（已在登录页或正在跳转时返回 false，交由调用方提示原文）
 */
function redirectToLogin(): boolean {
  const current = router.currentRoute.value;
  if (isRedirecting || current.path === "/login") return false;
  isRedirecting = true;

  ElMessage.warning("登录状态已失效，请重新登录");
  useUserStore().resetSession();
  router
    .replace({ path: "/login", query: { redirect: current.fullPath } })
    .finally(() => {
      isRedirecting = false;
    });
  return true;
}

function createOnResponseError() {
  return (error: any) => {
    if (error.response?.status === 401) {
      if (!redirectToLogin()) {
        ElMessage.error(error.response.data?.msg || "登录状态已失效");
      }
    } else if (error.response?.data) {
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
    onBizError: (code, msg) => {
      if (SESSION_INVALID_CODES.includes(code)) {
        if (redirectToLogin()) return;
      }
      ElMessage.error(msg || "系统出错");
    },
  });
}
