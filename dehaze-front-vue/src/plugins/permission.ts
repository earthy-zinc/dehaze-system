import router from "@/router";
import { usePermissionStore, useUserStore } from "@/store";
import NProgress from "@/utils/nprogress";
import { RouteRecordRaw } from "vue-router";

const NOT_FOUND_ROUTE_NAME = "NotFound";
// 已注册的动态路由顶层名称，菜单变更重注册前据此移除旧路由（移除父路由会级联移除其子路由）
let dynamicRouteNames: RouteRecordRaw["name"][] = [];

async function addDynamicRoutes(roles: string[]) {
  dynamicRouteNames.forEach((name) => {
    if (name && router.hasRoute(name)) {
      router.removeRoute(name);
    }
  });
  dynamicRouteNames = [];

  const permissionStore = usePermissionStore();
  const accessRoutes = await permissionStore.generateRoutes(roles);
  accessRoutes.forEach((route: RouteRecordRaw) => {
    router.addRoute(route);
    dynamicRouteNames.push(route.name);
  });
  // 动态路由加载后追加兜底，确保不存在的路径跳转 404 而非静默取消
  if (!router.hasRoute(NOT_FOUND_ROUTE_NAME)) {
    router.addRoute({
      path: "/:pathMatch(.*)*",
      name: NOT_FOUND_ROUTE_NAME,
      component: () => import("@/views/error-page/404.vue"),
      meta: { hidden: true },
    });
  }
}

/**
 * 菜单/角色权限变更后重建动态路由：移除旧路由并重新拉取注册，
 * 使新增菜单的路由立即可访问、已删除菜单的路由立即失效
 */
export async function reloadDynamicRoutes() {
  const userStore = useUserStore();
  await addDynamicRoutes(userStore.user.roles);
}

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
        try {
          await addDynamicRoutes(userStore.user.roles);
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
      try {
        const { roles } = await userStore.getUserInfo();
        await addDynamicRoutes(roles);
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
