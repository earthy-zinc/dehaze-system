/**
 * 认证 API
 *
 * 直接使用 dehaze-sdk-js 的 AuthAPI，不再维护独立定义。
 * 类型从 SDK 导出，函数包装保持调用方签名兼容。
 */

import { AuthAPI } from "dehaze-sdk-js";

export type {
  LoginData,
  LoginResult,
  LoginUser,
  AuthUserInfo,
  CaptchaResult,
  RefreshResult,
} from "dehaze-sdk-js";

// ==================== API 方法（函数包装） ====================

/** 获取验证码 */
export function getCaptcha() {
  return AuthAPI.getCaptcha();
}

/** 登录 */
export function login(data: import("dehaze-sdk-js").LoginData) {
  return AuthAPI.login(data);
}

/** 登出 */
export function logout() {
  return AuthAPI.logout();
}

/** 获取当前用户信息 */
export function getCurrentUser() {
  return AuthAPI.getCurrentUser();
}
