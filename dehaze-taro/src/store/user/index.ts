import { AuthAPI, UserAPI, TOKEN_KEY } from "dehaze-sdk-js";
import type { LoginData, UserInfo } from "dehaze-sdk-js";

import Taro from "@tarojs/taro";

// 定义用户状态类型
export interface UserState {
  user: UserInfo;
}

// 登录
export function login(loginData: LoginData) {
  return new Promise<void>((resolve, reject) => {
    AuthAPI.login(loginData)
      .then((data) => {
        const { tokenType, accessToken } = data;
        localStorage.setItem(TOKEN_KEY, tokenType + " " + accessToken); // Bearer eyJhbGciOiJIUzI1NiJ9.xxx.xxx
        resolve();
      })
      .catch((error) => {
        reject(error);
      });
  });
}

// 获取用户信息(用户昵称、头像、角色集合、权限集合)
export function getUserInfo() {
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
        resolve(data);
      })
      .catch((error) => {
        reject(error);
      });
  });
}

// 用户登出
export function logout() {
  return new Promise<void>((resolve, reject) => {
    AuthAPI.logout()
      .then(() => {
        localStorage.setItem(TOKEN_KEY, "");
        // 使用 Taro 的 API 进行页面跳转
        Taro.redirectTo({ url: "/pages/login/login" });
        resolve();
      })
      .catch((error) => {
        reject(error);
      });
  });
}

// 移除 token
export function resetToken() {
  return new Promise<void>((resolve) => {
    localStorage.setItem(TOKEN_KEY, "");
    resolve();
  });
}
