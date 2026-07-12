/**
 * 认证状态管理
 *
 * 管理：
 * - AccessToken / RefreshToken
 * - 当前用户信息
 * - 登录 / 登出 / 初始化逻辑
 */

import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { AuthUserInfo, CaptchaResult, LoginData, LoginResult } from "@/api/auth";
import { login as loginApi, logout as logoutApi, getCurrentUser, getCaptcha as getCaptchaApi } from "@/api/auth";
import { clearAuth as clearStorageAuth } from "@/api/request";
import { ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, USER_INFO_KEY } from "@/api/config";

export const useAuthStore = defineStore("auth", () => {
  // ==================== 状态 ====================

  /** AccessToken */
  const accessToken = ref<string>("");

  /** RefreshToken */
  const refreshToken = ref<string>("");

  /** 当前用户信息 */
  const userInfo = ref<AuthUserInfo | null>(null);

  /** 是否已登录 */
  const isLoggedIn = computed(() => !!accessToken.value);

  /** 用户名 */
  const username = computed(() => userInfo.value?.username || "");

  /** 昵称 */
  const nickname = computed(() => userInfo.value?.nickname || username.value);

  /** 用户ID */
  const userId = computed(() => userInfo.value?.userId || 0);

  /** 角色列表 */
  const roles = computed(() => userInfo.value?.roles || []);

  /** 权限列表 */
  const perms = computed(() => userInfo.value?.perms || []);

  // ==================== 方法 ====================

  /** 初始化：从 Storage 恢复登录态 */
  function init() {
    try {
      const token = uni.getStorageSync(ACCESS_TOKEN_KEY);
      const refresh = uni.getStorageSync(REFRESH_TOKEN_KEY);
      const userStr = uni.getStorageSync(USER_INFO_KEY);

      if (token) {
        accessToken.value = token;
      }
      if (refresh) {
        refreshToken.value = refresh;
      }
      if (userStr) {
        try {
          userInfo.value = JSON.parse(userStr);
        } catch {
          userInfo.value = null;
        }
      }
    } catch (error) {
      console.warn("[AuthStore] 初始化失败:", error);
    }
  }

  /** 登录 */
  async function login(data: LoginData): Promise<LoginResult> {
    const result = await loginApi(data);

    accessToken.value = result.accessToken;
    // 后端可能不返回 refreshToken，这里做兼容
    refreshToken.value = result.refreshToken || "";

    // 持久化 Token
    uni.setStorageSync(ACCESS_TOKEN_KEY, result.accessToken);
    if (result.refreshToken) {
      uni.setStorageSync(REFRESH_TOKEN_KEY, result.refreshToken);
    }

    // 登录成功后获取完整用户信息
    try {
      const user = await getCurrentUser();
      userInfo.value = user;
      uni.setStorageSync(USER_INFO_KEY, JSON.stringify(user));
    } catch {
      // 用户信息获取失败不影响登录流程
      console.warn("[AuthStore] 获取用户信息失败");
    }

    return result;
  }

  /** 登出 */
  async function logout(): Promise<void> {
    try {
      await logoutApi();
    } catch {
      // 登出 API 失败也清除本地状态
    }

    accessToken.value = "";
    refreshToken.value = "";
    userInfo.value = null;

    clearStorageAuth();
    uni.removeStorageSync(USER_INFO_KEY);
  }

  /** 获取验证码 */
  async function getCaptcha(): Promise<CaptchaResult> {
    return getCaptchaApi();
  }

  /** 检查是否有某权限 */
  function hasPerm(perm: string): boolean {
    if (!perms.value || perms.value.length === 0) return false;
    return perms.value.includes(perm);
  }

  /** 检查是否有任意角色 */
  function hasRole(role: string): boolean {
    if (!roles.value || roles.value.length === 0) return false;
    return roles.value.includes(role);
  }

  /** 是否为管理员 */
  function isAdmin(): boolean {
    // 后端返回的角色无 ROLE_ 前缀，兼容两种形式
    return hasRole("ROOT") || hasRole("ADMIN") || hasRole("ROLE_ROOT") || hasRole("ROLE_ADMIN");
  }

  return {
    // 状态
    accessToken,
    refreshToken,
    userInfo,
    isLoggedIn,
    username,
    nickname,
    userId,
    roles,
    perms,

    // 方法
    init,
    login,
    logout,
    getCaptcha,
    hasPerm,
    hasRole,
    isAdmin,
  };
});
