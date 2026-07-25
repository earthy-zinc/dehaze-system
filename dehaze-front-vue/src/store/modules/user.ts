import { resetRouter } from "@/router";
import { store } from "@/store";
import { clearAccessToken, setAccessToken } from "@/utils/auth";

import {
  AuthAPI,
  LoginData,
  UserAPI,
  UserInfo,
} from "dehaze-sdk-js";

export const useUserStore = defineStore("user", () => {
  const user = ref<UserInfo>({
    roles: [],
    perms: [],
  });

  /** 登录 */
  async function login(loginData: LoginData) {
    const { tokenType, accessToken } = await AuthAPI.login(loginData);
    const rememberMe = loginData.rememberMe !== false;
    setAccessToken(tokenType + " " + accessToken, rememberMe);
  }

  /** 刷新 accessToken（refreshToken 由 httpOnly Cookie 自动携带） */
  async function refreshAccessToken() {
    const { tokenType, accessToken } = await AuthAPI.refreshToken();
    const rememberMe = !!localStorage.getItem("rememberMe");
    setAccessToken(tokenType + " " + accessToken, rememberMe);
  }

  /** 获取用户信息（昵称、头像、角色、权限） */
  async function getUserInfo() {
    const data = await UserAPI.getInfo();
    if (!data.roles || data.roles.length === 0) {
      throw new Error("getUserInfo: roles must be a non-null array!");
    }
    Object.assign(user.value, { ...data });
    return data;
  }

  /** 登出 */
  async function logout() {
    await AuthAPI.logout();
    clearAccessToken();
    location.reload();
  }

  /** 清除 token 并重置路由 */
  function resetToken() {
    clearAccessToken();
    resetRouter();
  }

  return {
    user,
    login,
    refreshAccessToken,
    getUserInfo,
    logout,
    resetToken,
  };
});

// 非setup
export function useUserStoreHook() {
  return useUserStore(store);
}
