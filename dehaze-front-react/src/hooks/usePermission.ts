// usePermission.ts
import { TOKEN_KEY } from "@/enums/CacheEnum";
import { DisPatchType, RootState } from "@/store";
import { generateRoutes } from "@/store/modules/permissionSlice";
import { getUserInfo, resetToken } from "@/store/modules/userSlice";
import NProgress from "nprogress";
import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";

const whiteList = ["/login"];

export const usePermission = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const userStore = useSelector((state: RootState) => state.user);
  const permissionStore = useSelector((state: RootState) => state.permission);
  const dispatch = useDispatch<DisPatchType>();

  useEffect(() => {
    NProgress.start();
    const hasToken = localStorage.getItem(TOKEN_KEY);
    if (!hasToken && !whiteList.includes(location.pathname)) {
      navigate(`/login?redirect=${location.pathname}`, { replace: true });
      NProgress.done();
      return;
    }

    if (location.pathname === "/login") {
      navigate("/", { replace: true });
      NProgress.done();
      return;
    }

    const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;
    const findRoute = permissionStore.routes.find(
      (route) => route.path === location.pathname
    );

    if (hasRoles && !findRoute) {
      navigate("/404", { replace: true });
      NProgress.done();
      return;
    }

    if (!hasRoles) {
      try {
        dispatch(getUserInfo());
        dispatch(generateRoutes(userStore.user.roles));
        permissionStore.routes.forEach((route) => {
          // Add the route to your router configuration
          // You may want to use a state management solution to store these routes
        });
        navigate(location.pathname, { replace: true });
      } catch (error) {
        console.error(error);
        dispatch(resetToken());
        navigate(`/login?redirect=${location.pathname}`, {
          replace: true,
        });
      }
    }

    NProgress.done();
  }, [
    dispatch,
    location.pathname,
    navigate,
    permissionStore.routes,
    userStore.user.roles,
  ]);
};
