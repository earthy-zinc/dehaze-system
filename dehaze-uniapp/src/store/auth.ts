import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { AuthAPI, SESSION_KEY } from "dehaze-sdk-js";
import type {
  AuthUserInfo,
  LoginData,
  LoginResult,
  RegisterData,
} from "dehaze-sdk-js";
import { SESSION_INVALID_EVENT, USER_INFO_KEY } from "@/api/constants";

export const useAuthStore = defineStore("auth", () => {
  const sessionId = ref<string>("");
  const userInfo = ref<AuthUserInfo | null>(null);
  const isLoggedIn = computed(() => !!userInfo.value);
  const username = computed(() => userInfo.value?.username || "");
  const nickname = computed(() => userInfo.value?.nickname || username.value);
  const userId = computed(() => userInfo.value?.userId || 0);
  const roles = computed(() => userInfo.value?.roles || []);
  const perms = computed(() => userInfo.value?.perms || []);

  /** 持久化登录态到 storage */
  function persistSession(result: LoginResult) {
    sessionId.value = result.sessionId;
    uni.setStorageSync(SESSION_KEY, result.sessionId);
  }

  /** 登录/注册成功后拉取用户信息并持久化 */
  async function fetchUserInfo() {
    try {
      const user = await AuthAPI.getCurrentUser();
      userInfo.value = user;
      uni.setStorageSync(USER_INFO_KEY, JSON.stringify(user));
    } catch {
      // 用户信息拉取失败不阻塞进入首页，首页会以匿名态渲染或提示
      userInfo.value = null;
    }
  }

  /** 清空认证状态（内存 + storage），供主动登出与登录失效共用 */
  function clearAuthState() {
    sessionId.value = "";
    userInfo.value = null;
    uni.removeStorageSync(SESSION_KEY);
    uni.removeStorageSync(USER_INFO_KEY);
  }

  /** 从 storage 恢复登录态；并注册会话失效监听（防重） */
  function init() {
    uni.$off(SESSION_INVALID_EVENT);
    uni.$on(SESSION_INVALID_EVENT, clearAuthState);

    try {
      const sid = uni.getStorageSync(SESSION_KEY);
      const userStr = uni.getStorageSync(USER_INFO_KEY);

      if (sid) {
        sessionId.value = sid;
      }
      if (userStr) {
        try {
          userInfo.value = JSON.parse(userStr);
        } catch {
          userInfo.value = null;
        }
      }
    } catch {}
  }

  async function login(data: LoginData): Promise<LoginResult> {
    const result = await AuthAPI.login(data);
    persistSession(result);
    await fetchUserInfo();
    return result;
  }

  /** 注册成功后自动登录（后端直接返回会话），与登录共用写会话逻辑 */
  async function register(data: RegisterData): Promise<LoginResult> {
    const result = await AuthAPI.register(data);
    persistSession(result);
    await fetchUserInfo();
    return result;
  }

  async function logout(): Promise<void> {
    try {
      await AuthAPI.logout();
    } catch {}

    clearAuthState();
  }

  function hasPerm(perm: string): boolean {
    if (!perms.value || perms.value.length === 0) return false;
    return perms.value.includes(perm);
  }

  function hasRole(role: string): boolean {
    if (!roles.value || roles.value.length === 0) return false;
    return roles.value.includes(role);
  }

  return {
    sessionId,
    userInfo,
    isLoggedIn,
    username,
    nickname,
    userId,
    roles,
    perms,
    init,
    login,
    register,
    logout,
    clearAuthState,
    hasPerm,
    hasRole,
  };
});
