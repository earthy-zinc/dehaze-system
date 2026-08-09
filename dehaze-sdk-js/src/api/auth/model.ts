import { PageQuery } from "@/types";

export interface LoginData {
  username: string;
  password: string;
  captchaKey?: string;
  captchaCode?: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  username: string;
  password: string;
  nickname: string;
  captchaKey?: string;
  captchaCode?: string;
}

export interface LoginUser {
  id: number;
  username: string;
  nickname: string;
}

export interface LoginResult {
  sessionId: string;
  user: LoginUser;
}

export interface AuthUserInfo {
  userId: number;
  username: string;
  nickname: string;
  avatar?: string;
  roles: string[];
  perms: string[];
}

export interface CaptchaResult {
  captchaKey: string;
  captchaBase64: string;
}

export interface LoginLogQuery extends PageQuery {
  username?: string;
  ip?: string;
  status?: number;
  deviceType?: string;
  startTime?: string;
  endTime?: string;
}

export interface LoginLogVO {
  id: string;
  userId?: number;
  username: string;
  ip: string;
  location?: string;
  browser?: string;
  os?: string;
  status: number;
  message?: string;
  deviceType?: string;
  loginTime: string;
}

export interface SessionInfo {
  sessionId: string;
  username?: string;
  deviceType?: string;
  loginTime?: string;
  ip?: string;
  lastAccessTime?: string;
}
