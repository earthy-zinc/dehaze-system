// usePermission.ts
import { TOKEN_KEY } from "dehaze-sdk-js";
import { DisPatchType, RootState } from "@/store";
import { generateRoutes } from "@/store/modules/permissionSlice";
import { getUserInfo, resetToken } from "@/store/modules/userSlice";
import NProgress from "nprogress";
import { useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";

const whiteList = ["/login", "/403", "/404"];

/**
 * 路由守卫 + 用户信息初始化 Hook
 * 在 BasicLayout 中调用，确保：
 * 1. 未登录用户被重定向到 /login
 * 2. 登录后自动拉取用户信息（roles/permissions）与动态路由
 * 3. 已登录用户访问 /login 时重定向回首页
 */
export const usePermission = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const userStore = useSelector((state: RootState) => state.user);
  const dispatch = useDispatch<DisPatchType>();

  useEffect(() => {
    NProgress.start();
    const hasToken = localStorage.getItem(TOKEN_KEY);

    // 1. 未登录且不在白名单 -> 跳转登录页
    if (!hasToken && !whiteList.includes(location.pathname)) {
      navigate(`/login?redirect=${location.pathname}`, { replace: true });
      NProgress.done();
      return;
    }

    // 2. 已登录访问登录页 -> 跳转首页
    if (hasToken && location.pathname === "/login") {
      navigate("/", { replace: true });
      NProgress.done();
      return;
    }

    // 3. 已登录但未加载用户信息 -> 拉取用户信息并生成路由
    const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;
    if (hasToken && !hasRoles && !whiteList.includes(location.pathname)) {
      // 使用 async IIFE 避免在 effect 中直接返回 Promise
      (async () => {
        try {
          // 先拉取用户信息，拿到 roles 后再生成路由（避免竞态条件）
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
      })();
    }

    NProgress.done();
  }, [
    dispatch,
    location.pathname,
    navigate,
    userStore.user.roles,
  ]);
};

/**
 * 权限校验 Hook
 * 基于当前用户的角色与权限列表判断是否拥有指定权限
 * - ROOT 角色拥有所有权限
 * - 其他角色按 permissions 列表匹配
 * @returns hasPerm 函数，传入权限标识返回布尔值
 */
export const useHasPerm = () => {
  const userStore = useSelector((state: RootState) => state.user);
  const hasPerm = useCallback(
    (perm: string): boolean => {
      const roles = userStore.user.roles || [];
      // 超级管理员拥有所有按钮权限
      if (roles.includes("ROOT")) {
        return true;
      }
      const permissions = userStore.user.permissions || [];
      return permissions.includes(perm);
    },
    [userStore.user.roles, userStore.user.permissions]
  );
  return hasPerm;
};
