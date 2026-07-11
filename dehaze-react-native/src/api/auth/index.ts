import request from '@/utils/request';
import { CaptchaResult, LoginData, LoginResult } from './model';

class AuthAPI {
  /**
   * 登录API（JSON body）
   *
   * @param data {LoginData}
   */
  static login(data: LoginData) {
    return request<any, LoginResult>({
      url: '/api/v1/auth/login',
      method: 'post',
      data: data,
    });
  }

  /**
   * 注销API（POST）
   */
  static logout() {
    return request({
      url: '/api/v1/auth/logout',
      method: 'post',
    });
  }

  /**
   * 获取验证码
   */
  static getCaptcha() {
    return request<any, CaptchaResult>({
      url: '/api/v1/auth/captcha',
      method: 'get',
    });
  }
}

export default AuthAPI;
