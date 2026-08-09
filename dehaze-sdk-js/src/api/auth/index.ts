import request from "@/utils/request";
import { PageResult } from "@/types";
import {
  AuthUserInfo,
  CaptchaResult,
  LoginData,
  LoginLogQuery,
  LoginLogVO,
  LoginResult,
  RegisterData,
  SessionInfo,
} from "./model";

class AuthAPI {
  static login(data: LoginData) {
    return request<LoginResult>({
      url: "/api/v1/auth/login",
      method: "post",
      data: data,
    });
  }

  static register(data: RegisterData) {
    return request<LoginResult>({
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
    return request<AuthUserInfo>({
      url: "/api/v1/auth/me",
      method: "get",
    });
  }

  static getCaptcha() {
    return request<CaptchaResult>({
      url: "/api/v1/auth/captcha",
      method: "get",
    });
  }

  static getLoginLogs(query: LoginLogQuery) {
    return request<PageResult<LoginLogVO[]>>({
      url: "/api/v1/auth/login-logs",
      method: "get",
      params: query,
    });
  }

  static exportLoginLogs(query: LoginLogQuery) {
    return request<Blob>({
      url: "/api/v1/auth/login-logs/export",
      method: "get",
      params: query,
      responseType: "blob",
    });
  }

  static getSessions(username: string) {
    return request<SessionInfo[]>({
      url: "/api/v1/auth/sessions",
      method: "get",
      params: { username },
    });
  }

  static kickSession(sessionId: string) {
    return request({
      url: "/api/v1/auth/sessions/" + sessionId,
      method: "delete",
    });
  }
}

export default AuthAPI;
