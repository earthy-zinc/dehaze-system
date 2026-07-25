import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { AuthAPI, SESSION_KEY } from "dehaze-sdk-js";
import type {
  AuthUserInfo,
  CaptchaResult,
  LoginData,
  LoginResult,
} from "dehaze-sdk-js";
import { clearAuth as clearStorageAuth } from "@/api/sdk-setup";
import { USER_INFO_KEY } from "@/api/config";

export const useAuthStore = defineStore("auth", () => {
  const sessionId = ref<string>("");
  const userInfo = ref<AuthUserInfo | null>(null);
  const isLoggedIn = computed(() => !!userInfo.value);
  const username = computed(() => userInfo.value?.username || "");
  const nickname = computed(() => userInfo.value?.nickname || username.value);
  const userId = computed(() => userInfo.value?.userId || 0);
  const roles = computed(() => userInfo.value?.roles || []);
  const perms = computed(() => userInfo.value?.perms || []);

  function init() {
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

    sessionId.value = result.sessionId;
    uni.setStorageSync(SESSION_KEY, result.sessionId);

    try {
      const user = await AuthAPI.getCurrentUser();
      userInfo.value = user;
      uni.setStorageSync(USER_INFO_KEY, JSON.stringify(user));
    } catch {}

    return result;
  }

  async function logout(): Promise<void> {
    try {
      await AuthAPI.logout();
    } catch {}

    sessionId.value = "";
    userInfo.value = null;
    clearStorageAuth();
  }

  async function getCaptcha(): Promise<CaptchaResult> {
    return AuthAPI.getCaptcha();
  }

  function hasPerm(perm: string): boolean {
    if (!perms.value || perms.value.length === 0) return false;
    return perms.value.includes(perm);
  }

  function hasRole(role: string): boolean {
    if (!roles.value || roles.value.length === 0) return false;
    return roles.value.includes(role);
  }

  function isAdmin(): boolean {
    return (
      hasRole("ROOT") ||
      hasRole("ADMIN") ||
      hasRole("ROLE_ROOT") ||
      hasRole("ROLE_ADMIN")
    );
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
    logout,
    getCaptcha,
    hasPerm,
    hasRole,
    isAdmin,
  };
});
