import { DisPatchType, RootState } from "@/store";
import { generateRoutes } from "@/store/modules/permissionSlice";
import { getUserInfo, resetToken } from "@/store/modules/userSlice";
import NProgress from "nprogress";
import { useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";

const whiteList = ["/login", "/register", "/403", "/404"];

export const usePermission = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const userStore = useSelector((state: RootState) => state.user);
  const permissionRoutes = useSelector(
    (state: RootState) => state.permission.routes
  );
  const dispatch = useDispatch<DisPatchType>();

  useEffect(() => {
    NProgress.start();

    (async () => {
      const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;

      if (location.pathname === "/login") {
        if (hasRoles) {
          navigate("/", { replace: true });
        }
        NProgress.done();
        return;
      }

      if (whiteList.includes(location.pathname)) {
        NProgress.done();
        return;
      }

      if (!hasRoles || permissionRoutes.length === 0) {
        try {
          let roles = userStore.user.roles || [];
          if (!hasRoles) {
            const resultAction = await dispatch(getUserInfo());
            if (!getUserInfo.fulfilled.match(resultAction)) {
              throw new Error("获取用户信息失败");
            }
            roles = resultAction.payload.roles || [];
          }
          await dispatch(generateRoutes(roles));
        } catch {
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
    permissionRoutes.length,
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
