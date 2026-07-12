import Taro from '@tarojs/taro';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';
import {
  configJavaAxios,
  configPythonAxios,
  ResponseData,
  ResultEnum,
} from 'dehaze-sdk-js';
import { storage } from '@/utils/storage';
import { apiConfig } from '@/config/api';

/** 判断错误码前缀 */
function codeStartsWith(code: string | undefined, prefix: string): boolean {
  return !!code && code.startsWith(prefix);
}

/** 跳转登录页（同步清除本地认证信息后跳转，防重复跳转） */
let isRedirecting = false;
function redirectToLogin(): void {
  if (isRedirecting) return;
  isRedirecting = true;

  // 如果当前已经在登录页，不再跳转
  const pages = Taro.getCurrentPages();
  const currentRoute = pages.length > 0 ? '/' + (pages[pages.length - 1].route || '') : '';
  if (currentRoute === '/pages/login/index') {
    isRedirecting = false;
    return;
  }

  storage.removeToken();
  Taro.removeStorageSync('userInfo');
  Taro.removeStorageSync('permissions');
  Taro.removeStorageSync('roles');

  Taro.redirectTo({
    url: '/pages/login/index',
    complete: () => { isRedirecting = false; }
  });
}

/**
 * 响应错误处理：按错误码段分类处理
 *
 * - A02x: 用户/登录异常 → 登录失效的跳转登录页，其余提示
 * - A03x: 权限异常 → 提示无权限，未授权跳转登录
 * - A04x: 参数错误 → 提示
 * - Bxxx: 系统错误 → 提示
 */
function onResponseError(error: AxiosError): void {
  if (error.response?.data) {
    const { code, msg } = error.response.data as ResponseData;

    // A02x — 用户登录异常
    if (codeStartsWith(code, 'A02')) {
      // token 失效 / 客户端认证失败 / 账户异常 → 跳转登录
      if (
        code === ResultEnum.TOKEN_INVALID ||
        code === ResultEnum.TOKEN_ACCESS_FORBIDDEN ||
        code === ResultEnum.CLIENT_AUTHENTICATION_FAILED ||
        code === ResultEnum.USER_NOT_EXIST ||
        code === ResultEnum.USER_ACCOUNT_LOCKED ||
        code === ResultEnum.USER_ACCOUNT_INVALID
      ) {
        Taro.showModal({
          title: '提示',
          content: msg || '登录已失效，请重新登录',
          showCancel: false,
          confirmText: '重新登录',
          success: (res) => {
            if (res.confirm) {
              redirectToLogin();
            }
          },
        });
        return;
      }
      // 其余 A02x（密码错误、验证码等）→ 仅提示
      Taro.showToast({ title: msg || '登录异常', icon: 'none' });
      return;
    }

    // A03x — 访问权限异常
    if (codeStartsWith(code, 'A03')) {
      if (code === ResultEnum.ACCESS_UNAUTHORIZED) {
        redirectToLogin();
        return;
      }
      Taro.showToast({ title: msg || '无访问权限', icon: 'none' });
      return;
    }

    // A04x — 请求参数错误
    if (codeStartsWith(code, 'A04')) {
      Taro.showToast({ title: msg || '请求参数错误', icon: 'none' });
      return;
    }

    // Bxxx — 系统错误
    if (codeStartsWith(code, 'B')) {
      Taro.showToast({ title: msg || '系统异常，请稍后重试', icon: 'none' });
      return;
    }

    // 其他业务错误
    Taro.showToast({ title: msg || '系统出错啦', icon: 'none' });
    return;
  }

  // 无响应（网络断开、超时、CORS）
  if (error.request) {
    Taro.showToast({ title: '网络异常，请检查网络连接', icon: 'none' });
    return;
  }

  Taro.showToast({ title: error.message || '请求发送失败', icon: 'none' });
}

export default function configRequest(): void {
  configJavaAxios({
    getToken: () => storage.getToken(),
    onRequest: (config: InternalAxiosRequestConfig) => ({
      ...config,
      baseURL: apiConfig.java,
    }),
    onResponseError,
  });

  configPythonAxios({
    getToken: () => storage.getToken(),
    onRequest: (config: InternalAxiosRequestConfig) => ({
      ...config,
      baseURL: apiConfig.python,
    }),
    onResponseError,
  });
}
