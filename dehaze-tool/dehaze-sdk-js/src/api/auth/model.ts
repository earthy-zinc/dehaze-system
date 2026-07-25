/**
 * 登录请求参数
 */
export interface LoginData {
  /** 用户名 */
  username: string;
  /** 密码 */
  password: string;
  /** 验证码缓存key */
  captchaKey?: string;
  /** 验证码 */
  captchaCode?: string;
  /** 记住我（前端控制 Token 存储策略，不发送给后端） */
  rememberMe?: boolean;
}

/**
 * 登录响应中的用户基本信息
 */
export interface LoginUser {
  id: number;
  username: string;
  nickname: string;
}

/**
 * 登录响应
 */
export interface LoginResult {
  accessToken: string;
  tokenType: string;
  refreshToken: string;
  expires: number;
  user: LoginUser;
}

/**
 * 当前用户信息（/auth/me 响应）
 */
export interface AuthUserInfo {
  /** 用户ID */
  userId: number;
  /** 用户名 */
  username: string;
  /** 昵称 */
  nickname: string;
  /** 头像 */
  avatar?: string;
  /** 角色列表（含 ROLE_ 前缀） */
  roles: string[];
  /** 权限列表 */
  perms: string[];
}

/**
 * 验证码响应
 */
export interface CaptchaResult {
  /** 验证码缓存key */
  captchaKey: string;
  /** 验证码图片Base64字符串 */
  captchaBase64: string;
}

/**
 * 刷新Token响应（与登录响应结构相同）
 */
export type RefreshResult = LoginResult;
