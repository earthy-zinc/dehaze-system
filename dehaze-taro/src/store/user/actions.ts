import type { UserInfo } from "dehaze-sdk-js";

export const SET_USER_INFO = "SET_USER_INFO";
export const RESET_USER_INFO = "RESET_USER_INFO";

export const setUserInfo = (user: UserInfo) => ({
  type: SET_USER_INFO,
  payload: user,
});

export const resetUserInfo = () => ({
  type: RESET_USER_INFO,
});
