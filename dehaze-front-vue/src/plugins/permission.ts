import router from "@/router";
import { usePermissionStore, useUserStore } from "@/store";
import NProgress from "@/utils/nprogress";
import { RouteRecordRaw } from "vue-router";

export function setupPermission() {
  const whiteList = ["/login"];

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
      if (to.matched.length === 0) {
        from.name ? next({ name: from.name }) : next("/404");
      } else {
        next();
      }
    } else {
      const permissionStore = usePermissionStore();
      try {
        const { roles } = await userStore.getUserInfo();
        const accessRoutes = await permissionStore.generateRoutes(roles);
        accessRoutes.forEach((route: RouteRecordRaw) => {
          router.addRoute(route);
        });
        next({ ...to, replace: true });
      } catch {
        await userStore.resetToken();
        next(`/login?redirect=${to.path}`);
        NProgress.done();
      }
    }
  });

  router.afterEach(() => {
    NProgress.done();
  });
}
