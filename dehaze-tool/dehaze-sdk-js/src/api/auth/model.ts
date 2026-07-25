export interface LoginData {
  username: string;
  password: string;
  captchaKey?: string;
  captchaCode?: string;
  rememberMe?: boolean;
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
