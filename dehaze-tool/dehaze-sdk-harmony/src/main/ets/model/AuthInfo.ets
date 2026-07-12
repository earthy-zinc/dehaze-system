/**
 * 认证相关数据模型
 *
 * 与后端 dto.LoginResult / dto.CaptchaResult / vo.UserInfoVO 对齐
 */

/** 登录请求参数 */
export class LoginData {
  /** 用户名 */
  username: string = '';
  /** 密码 */
  password: string = '';
  /** 验证码缓存 key */
  captchaKey?: string = undefined;
  /** 验证码 */
  captchaCode?: string = undefined;
}

/** 登录返回的用户基本信息 */
export class LoginUser {
  id: number = 0;
  username: string = '';
  nickname: string = '';
}

/** 登录响应（后端 dto.LoginResult） */
export class LoginResult {
  /** 访问 token */
  accessToken: string = '';
  /** token 类型 */
  tokenType: string = '';
  /** 刷新 token */
  refreshToken: string = '';
  /** 过期时间（毫秒） */
  expires: number = 0;
  /** 用户基本信息 */
  user?: LoginUser = undefined;
}

/** 验证码响应（后端 dto.CaptchaResult） */
export class CaptchaResult {
  /** 验证码缓存 key */
  captchaKey: string = '';
  /** 验证码图片 Base64 字符串 */
  captchaBase64: string = '';
}

/**
 * 当前登录用户信息（后端 vo.UserInfoVO，/auth/me 响应）
 */
export class AuthUserInfo {
  /** 用户 ID */
  userId: number = 0;
  /** 用户名 */
  username: string = '';
  /** 昵称 */
  nickname: string = '';
  /** 头像地址 */
  avatar?: string = undefined;
  /** 角色编码集合 */
  roles: string[] = [];
  /** 权限标识集合 */
  perms: string[] = [];
}
