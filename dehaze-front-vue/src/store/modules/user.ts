import { resetRouter } from "@/router";
import { store } from "@/store";

import { AuthAPI, LoginData, UserAPI, UserInfo } from "dehaze-sdk-js";

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

  function resetToken() {
    resetRouter();
  }

  return {
    user,
    login,
    getUserInfo,
    logout,
    resetToken,
  };
});

export function useUserStoreHook() {
  return useUserStore(store);
}
