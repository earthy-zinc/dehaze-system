import { store } from "@/store";

import { AuthAPI, LoginData, UserAPI, UserInfo } from "clarity-sdk-js";

export const useUserStore = defineStore("user", () => {
  const user = ref<UserInfo>({
    roles: [],
    perms: [],
  });

  async function login(loginData: LoginData) {
    await AuthAPI.login(loginData);
  }

  async function getUserInfo() {
    const data = await UserAPI.getInfo();
    if (!data.roles || data.roles.length === 0) {
      throw new Error("getUserInfo: roles must be a non-null array!");
    }
    Object.assign(user.value, { ...data });
    return data;
  }

  async function logout() {
    await AuthAPI.logout();
    location.reload();
  }

  /**
   * 会话失效兜底：内存 roles 是「已登录」的唯一凭证（cookie 为 HttpOnly，JS 不可清），
   * 必须连用户信息一并清空，登录页才不会被守卫弹回首页。
   */
  function resetSession() {
    user.value = { roles: [], perms: [] };
  }

  return {
    user,
    login,
    getUserInfo,
    logout,
    resetSession,
  };
});

export function useUserStoreHook() {
  return useUserStore(store);
}
