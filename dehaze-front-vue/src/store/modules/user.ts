import { REFRESH_TOKEN_KEY, TOKEN_KEY } from "@/enums/CacheEnum";
import { resetRouter } from "@/router";
import { store } from "@/store";

import { AuthAPI, LoginData, UserAPI, UserInfo } from "dehaze-sdk-js";

export const useUserStore = defineStore("user", () => {
  const user = ref<UserInfo>({
    roles: [],
    perms: [],
  });

  /**
   * 登录
   *
   * @param {LoginData}
   * @returns
   */
  function login(loginData: LoginData) {
    return new Promise<void>((resolve, reject) => {
      AuthAPI.login(loginData)
        .then((data) => {
          const { tokenType, accessToken, refreshToken } = data;
          localStorage.setItem(TOKEN_KEY, tokenType + " " + accessToken); // Bearer eyJhbGciOiJIUzI1NiJ9.xxx.xxx
          localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
          resolve();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  /**
   * 刷新 accessToken
   *
   * @param refreshToken 刷新令牌
   */
  function refreshAccessToken(refreshToken: string) {
    return new Promise<void>((resolve, reject) => {
      AuthAPI.refreshToken(refreshToken)
        .then((data) => {
          const {
            tokenType,
            accessToken,
            refreshToken: newRefreshToken,
          } = data;
          localStorage.setItem(TOKEN_KEY, tokenType + " " + accessToken);
          localStorage.setItem(REFRESH_TOKEN_KEY, newRefreshToken);
          resolve();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  // 获取信息(用户昵称、头像、角色集合、权限集合)
  function getUserInfo() {
    return new Promise<UserInfo>((resolve, reject) => {
      UserAPI.getInfo()
        .then((data) => {
          if (!data) {
            reject("Verification failed, please Login again.");
            return;
          }
          if (!data.roles || data.roles.length <= 0) {
            reject("getUserInfo: roles must be a non-null array!");
            return;
          }
          Object.assign(user.value, { ...data });
          resolve(data);
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  // user logout
  function logout() {
    return new Promise<void>((resolve, reject) => {
      AuthAPI.logout()
        .then(() => {
          localStorage.setItem(TOKEN_KEY, "");
          localStorage.setItem(REFRESH_TOKEN_KEY, "");
          location.reload(); // 清空路由
          resolve();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  // remove token
  function resetToken() {
    return new Promise<void>((resolve) => {
      localStorage.setItem(TOKEN_KEY, "");
      localStorage.setItem(REFRESH_TOKEN_KEY, "");
      resetRouter();
      resolve();
    });
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
