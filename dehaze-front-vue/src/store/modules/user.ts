import { resetRouter } from "@/router";
import { store } from "@/store";

import {
  AuthAPI,
  LoginData,
  REFRESH_TOKEN_KEY,
  TOKEN_KEY,
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
    const { tokenType, accessToken, refreshToken } =
      await AuthAPI.login(loginData);
    localStorage.setItem(TOKEN_KEY, tokenType + " " + accessToken);
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  }

  /** 刷新 accessToken */
  async function refreshAccessToken(refreshToken: string) {
    const {
      tokenType,
      accessToken,
      refreshToken: newRefreshToken,
    } = await AuthAPI.refreshToken(refreshToken);
    localStorage.setItem(TOKEN_KEY, tokenType + " " + accessToken);
    localStorage.setItem(REFRESH_TOKEN_KEY, newRefreshToken);
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
    localStorage.setItem(TOKEN_KEY, "");
    localStorage.setItem(REFRESH_TOKEN_KEY, "");
    location.reload();
  }

  /** 清除 token 并重置路由 */
  function resetToken() {
    localStorage.setItem(TOKEN_KEY, "");
    localStorage.setItem(REFRESH_TOKEN_KEY, "");
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
