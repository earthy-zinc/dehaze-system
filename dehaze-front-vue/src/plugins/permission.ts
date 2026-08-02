import router from "@/router";
import { usePermissionStore, useUserStore } from "@/store";
import NProgress from "@/utils/nprogress";
import { RouteRecordRaw } from "vue-router";

export function setupPermission() {
  const whiteList = ["/login", "/register"];
  let isDynamicRoutesAdded = false;

  router.beforeEach(async (to, from, next) => {
    NProgress.start();

    const userStore = useUserStore();
    const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;

    if (to.path === "/login") {
      if (hasRoles) {
        next({ path: "/" });
      } else {
        next();
      }
      NProgress.done();
      return;
    }

    if (whiteList.includes(to.path)) {
      next();
      NProgress.done();
      return;
    }

    if (hasRoles) {
      if (!isDynamicRoutesAdded) {
        const permissionStore = usePermissionStore();
        try {
          const accessRoutes = await permissionStore.generateRoutes(
            userStore.user.roles
          );
          accessRoutes.forEach((route: RouteRecordRaw) => {
            router.addRoute(route);
          });
          // 动态路由加载后追加兜底，确保不存在的路径跳转 404 而非静默取消
          router.addRoute({
            path: "/:pathMatch(.*)*",
            component: () => import("@/views/error-page/404.vue"),
            meta: { hidden: true },
          });
          isDynamicRoutesAdded = true;
        } catch (e) {
          userStore.resetToken();
          next(`/login?redirect=${to.path}`);
          NProgress.done();
          return;
        }
        next({ ...to, replace: true });
        return;
      }
      next();
    } else {
      const permissionStore = usePermissionStore();
      try {
        const { roles } = await userStore.getUserInfo();
        const accessRoutes = await permissionStore.generateRoutes(roles);
        accessRoutes.forEach((route: RouteRecordRaw) => {
          router.addRoute(route);
        });
        router.addRoute({
          path: "/:pathMatch(.*)*",
          component: () => import("@/views/error-page/404.vue"),
          meta: { hidden: true },
        });
        isDynamicRoutesAdded = true;
        next({ ...to, replace: true });
      } catch (e) {
        userStore.resetToken();
        next(`/login?redirect=${to.path}`);
        NProgress.done();
      }
    }
  });

  router.afterEach(() => {
    NProgress.done();
  });
}
