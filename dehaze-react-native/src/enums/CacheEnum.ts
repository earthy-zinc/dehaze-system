/**
 * 本地缓存 Key
 */
export const CacheEnum = {
  /** 访问令牌 */
  TOKEN: 'accessToken',
  /** 刷新令牌 */
  REFRESH_TOKEN: 'refreshToken',
  /** 当前登录用户权限信息 */
  AUTH_INFO: 'authInfo',
} as const;
