/**
 * 认证 API
 *
 * API 路径与 dehaze-sdk-js 及后端路由保持一致：
 * - GET  /auth/captcha  获取验证码
 * - POST /auth/login     登录
 * - POST /auth/logout    登出
 * - GET  /auth/me        获取当前用户信息
 * - POST /auth/refresh   刷新 Token
 */

import { get, post } from "./request";

// ==================== 类型定义 ====================

/** 登录请求参数 */
export interface LoginData {
  username: string;
  password: string;
  captchaKey?: string;
  captchaCode?: string;
}

/** 登录响应中的用户基本信息 */
export interface LoginUser {
  id: number;
  username: string;
  nickname: string;
}

/** 登录响应 */
export interface LoginResult {
  accessToken: string;
  tokenType: string;
  /** 后端当前未返回，保留为可选 */
  refreshToken?: string;
  /** 后端当前未返回，保留为可选 */
  expires?: number;
  user: LoginUser;
}

/** 当前用户信息（/auth/me 响应） */
export interface AuthUserInfo {
  userId: number;
  username: string;
  nickname?: string;
  avatar?: string;
  /** 后端返回的角色列表（无 ROLE_ 前缀，如 ["ROOT","ADMIN"]） */
  roles: string[];
  /** 后端返回的权限列表（字段名为 permissions） */
  perms: string[];
}

/** 验证码响应 */
export interface CaptchaResult {
  /** 验证码缓存 key */
  captchaKey: string;
  /** 验证码图片 Base64 */
  captchaBase64: string;
}

/** 刷新 Token 响应 */
export type RefreshResult = LoginResult;

// ==================== API 方法 ====================

/** 获取验证码 */
export function getCaptcha() {
  return get<CaptchaResult>("/auth/captcha");
}

/** 登录 */
export function login(data: LoginData) {
  return post<LoginResult>(
    "/auth/login",
    data as unknown as Record<string, unknown>
  );
}

/** 登出 */
export function logout() {
  return post("/auth/logout");
}

/** 获取当前用户信息 */
export function getCurrentUser() {
  // 后端 /auth/me 返回 { userId, username, nickname, roles, permissions }
  // 这里映射为前端统一类型 AuthUserInfo（perms 字段）
  return get<Omit<AuthUserInfo, "perms"> & { permissions: string[] }>(
    "/auth/me"
  ).then((data) => ({
    userId: data.userId,
    username: data.username,
    nickname: data.nickname,
    avatar: data.avatar,
    roles: data.roles || [],
    perms: data.permissions || [],
  }));
}

/** 刷新 Token */
export function refreshTokenApi(refreshToken: string) {
  return post<RefreshResult>("/auth/refresh", { refreshToken });
}
