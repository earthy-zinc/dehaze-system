import type { UserState } from ".";
import type { UserInfo } from "dehaze-sdk-js";

// 定义 RootState 类型
interface RootState {
  user: UserState;
}

// 获取用户信息的选择器
export const selectUserInfo = (state: RootState): UserInfo => state.user.user;

// 检查用户是否有特定角色的选择器
export const selectHasRole = (state: RootState, role: string): boolean => {
  return state.user.user.roles?.includes(role) || false;
};

// 检查用户是否有特定权限的选择器
export const selectHasPermission = (
  state: RootState,
  permission: string
): boolean => {
  return state.user.user.perms?.includes(permission) || false;
};

// 检查用户是否已登录的选择器
export const selectIsLoggedIn = (state: RootState): boolean => {
  const token = localStorage.getItem("accessToken");
  return !!token;
};
