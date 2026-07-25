import request from "@/utils/request";
import { AuthUserInfo, CaptchaResult, LoginData, LoginResult, RegisterData } from "./model";

class AuthAPI {
  static login(data: LoginData) {
    return request<any, LoginResult>({
      url: "/api/v1/auth/login",
      method: "post",
      data: data,
    });
  }

  static register(data: RegisterData) {
    return request<any, LoginResult>({
      url: "/api/v1/auth/register",
      method: "post",
      data: data,
    });
  }

  static logout() {
    return request({
      url: "/api/v1/auth/logout",
      method: "post",
    });
  }

  static getCurrentUser() {
    return request<any, AuthUserInfo>({
      url: "/api/v1/auth/me",
      method: "get",
    });
  }

  static getCaptcha() {
    return request<any, CaptchaResult>({
      url: "/api/v1/auth/captcha",
      method: "get",
    });
  }
}

export default AuthAPI;
