import { getAccessToken } from "@/utils/auth";
import { DisPatchType, RootState } from "@/store";
import { generateRoutes } from "@/store/modules/permissionSlice";
import {
  getUserInfo,
  refreshAccessToken,
  resetToken,
} from "@/store/modules/userSlice";
import NProgress from "nprogress";
import { useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";

const whiteList = ["/login", "/403", "/404"];

export const usePermission = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const userStore = useSelector((state: RootState) => state.user);
  const dispatch = useDispatch<DisPatchType>();

  useEffect(() => {
    NProgress.start();

    (async () => {
      let hasToken = getAccessToken();

      // accessToken 不存在，尝试用 httpOnly Cookie 中的 refreshToken 刷新
      if (!hasToken) {
        try {
          await dispatch(refreshAccessToken()).unwrap();
          hasToken = getAccessToken();
        } catch {
          dispatch(resetToken());
        }
      }

      if (!hasToken && !whiteList.includes(location.pathname)) {
        navigate(`/login?redirect=${location.pathname}`, { replace: true });
        NProgress.done();
        return;
      }

      if (hasToken && location.pathname === "/login") {
        navigate("/", { replace: true });
        NProgress.done();
        return;
      }

      const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;
      if (hasToken && !hasRoles && !whiteList.includes(location.pathname)) {
        try {
          const resultAction = await dispatch(getUserInfo());
          if (!getUserInfo.fulfilled.match(resultAction)) {
            throw new Error("获取用户信息失败");
          }
          const roles = resultAction.payload.roles || [];
          await dispatch(generateRoutes(roles));
        } catch (error) {
          console.error("初始化用户信息失败:", error);
          dispatch(resetToken());
          navigate(`/login?redirect=${location.pathname}`, {
            replace: true,
          });
        }
      }

      NProgress.done();
    })();
  }, [
    dispatch,
    location.pathname,
    navigate,
    userStore.user.roles,
  ]);
};

export const useHasPerm = () => {
  const userStore = useSelector((state: RootState) => state.user);
  const hasPerm = useCallback(
    (perm: string): boolean => {
      const roles = userStore.user.roles || [];
      if (roles.includes("ROOT")) {
        return true;
      }
      const perms = userStore.user.perms || [];
      return perms.includes(perm);
    },
    [userStore.user.roles, userStore.user.perms]
  );
  return hasPerm;
};
