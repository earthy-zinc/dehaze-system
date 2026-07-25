import request from "@/utils/request";
import { AuthUserInfo, CaptchaResult, LoginData, LoginResult, RefreshResult } from "./model";

class AuthAPI {
  /**
   * 登录API（JSON body）
   *
   * @param data {LoginData}
   */
  static login(data: LoginData) {
    return request<any, LoginResult>({
      url: "/api/v1/auth/login",
      method: "post",
      data: data,
    });
  }

  /**
   * 注销API（POST）
   */
  static logout() {
    return request({
      url: "/api/v1/auth/logout",
      method: "post",
    });
  }

  /**
   * 获取当前用户信息
   */
  static getCurrentUser() {
    return request<any, AuthUserInfo>({
      url: "/api/v1/auth/me",
      method: "get",
    });
  }

  /**
   * 刷新Token（使用当前请求的 refreshToken）
   */
  static refreshToken(refreshToken?: string) {
    return request<any, RefreshResult>({
      url: "/api/v1/auth/refresh",
      method: "post",
      data: refreshToken ? { refreshToken } : {},
    });
  }

  /**
   * 获取验证码
   */
  static getCaptcha() {
    return request<any, CaptchaResult>({
      url: "/api/v1/auth/captcha",
      method: "get",
    });
  }
}

export default AuthAPI;
