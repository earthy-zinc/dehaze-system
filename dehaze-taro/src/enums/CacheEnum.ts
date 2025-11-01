/**
 * 缓存键枚举
 */
export enum CacheEnum {
  // 认证相关
  TOKEN = 'token',
  USER_INFO = 'userInfo',
  PERMISSIONS = 'permissions',
  ROLES = 'roles',

  // 系统配置
  SELECTED_DEPT = 'selectedDept',
  CACHE_EXPIRE = 'cacheExpire',

  // 主题设置
  THEME = 'theme',
  LANGUAGE = 'language',

  // 应用配置
  APP_CONFIG = 'appConfig',
  USER_CONFIG = 'userConfig',
}

/**
 * Token 存储键
 */
export const TOKEN_KEY = CacheEnum.TOKEN;

/**
 * 缓存过期时间（毫秒）
 */
export const CACHE_EXPIRE_TIME = {
  TOKEN: 1000 * 60 * 60 * 24, // 24小时
  USER_INFO: 1000 * 60 * 60 * 2, // 2小时
  PERMISSIONS: 1000 * 60 * 60 * 2, // 2小时
  CONFIG: 1000 * 60 * 60 * 24, // 24小时
};