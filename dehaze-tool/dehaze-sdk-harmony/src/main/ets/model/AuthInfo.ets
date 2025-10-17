/**
 * 登录请求参数
 */
export class LoginData {
  /**
   * 用户名
   */
  username: string = '';
  /**
   * 密码
   */
  password: string = '';
  
  /**
   * 验证码缓存key
   */
  captchaKey?: string = undefined;
  
  /**
   * 验证码
   */
  captchaCode?: string = undefined;
}

/**
 * 登录响应
 */
export class LoginResult {
  /**
   * 访问token
   */
  accessToken?: string = undefined;
  /**
   * 过期时间(单位：毫秒)
   */
  expires?: number = undefined;
  /**
   * 刷新token
   */
  refreshToken?: string = undefined;
  /**
   * token 类型
   */
  tokenType?: string = undefined;
}

/**
 * 验证码响应
 */
export class CaptchaResult {
  /**
   * 验证码缓存key
   */
  captchaKey: string = '';
  /**
   * 验证码图片Base64字符串
   */
  captchaBase64: string = '';
}