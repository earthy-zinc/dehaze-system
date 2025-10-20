import { useDispatch, useSelector } from "react-redux";
import type { UserState } from ".";
import type { UserInfo } from "dehaze-sdk-js";
import { resetUserInfo, setUserInfo } from "./actions";

// 定义 RootState 类型
interface RootState {
  user: UserState;
}

// 获取用户信息的 hook
export const useUserInfo = () => {
  return useSelector((state: RootState) => state.user.user);
};

// 设置用户信息的 hook
export const useUserActions = () => {
  const dispatch = useDispatch();

  const setUserInfoAction = (user: UserInfo) => {
    dispatch(setUserInfo(user));
  };

  const resetUserInfoAction = () => {
    dispatch(resetUserInfo());
  };

  return {
    setUserInfo: setUserInfoAction,
    resetUserInfo: resetUserInfoAction,
  };
};
