import { SESSION_KEY } from "dehaze-sdk-js";
import { SESSION_INVALID_EVENT, USER_INFO_KEY } from "./constants";
import { LOGIN_PATH } from "@/routers/guard";

let isRedirecting = false;

/**
 * 会话失效统一处理：清本地认证态、广播事件通知 auth store、重定向到登录页。
 * 供 axios 响应拦截器与不走 axios 的上传链路（uni.uploadFile）共用。
 */
export function redirectToLogin() {
  if (isRedirecting) return;
  isRedirecting = true;
  // 清理本地认证态（storage + 内存态由 auth store 监听事件同步清空）
  uni.removeStorageSync(SESSION_KEY);
  uni.removeStorageSync(USER_INFO_KEY);
  uni.$emit(SESSION_INVALID_EVENT);
  uni.reLaunch({
    url: LOGIN_PATH,
    complete: () => {
      isRedirecting = false;
    },
  });
}
