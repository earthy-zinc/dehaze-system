import router from "@/router";
import { usePermissionStore, useUserStore } from "@/store";
import NProgress from "@/utils/nprogress";
import { RouteRecordRaw } from "vue-router";

const NOT_FOUND_ROUTE_NAME = "NotFound";
// 已注册的动态路由顶层名称，菜单变更重注册前据此移除旧路由（移除父路由会级联移除其子路由）
let dynamicRouteNames: RouteRecordRaw["name"][] = [];
// 动态路由是否已注册（冷启动预注册与路由守卫共用同一标志）
let isDynamicRoutesAdded = false;

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

/**
 * 冷启动预注册动态路由：必须在 `app.use(router)`（首帧解析）之前调用。
 *
 * 动态路由来自会话用户信息（cookie 会话，roles 不落 localStorage）。不预注册时，
 * 首帧解析深链（如 `/#/admin/ai-billing/overview`）时路由表里还没有这些路由，
 * 会先打一条 `[Vue Router warn]: No match found for location with path "..."`，
 * 再由守卫补注册重导航兜底——用户看到一次首页闪烁、控制台留下告警（评审已发现）。
 * 未登录/接口失败不阻塞启动，交由路由守卫跳登录。
 */
export async function preloadDynamicRoutes(): Promise<void> {
  if (isDynamicRoutesAdded) return;
  const userStore = useUserStore();
  try {
    const { roles } = await userStore.getUserInfo();
    await addDynamicRoutes(roles);
    isDynamicRoutesAdded = true;
  } catch {
    // 未登录或会话失效：维持未注册，交由路由守卫处理
  }
}

export function setupPermission() {
  const whiteList = ["/login", "/register"];

  router.beforeEach(async (to, from, next) => {
    NProgress.start();

    const userStore = useUserStore();
    const hasRoles = userStore.user.roles && userStore.user.roles.length > 0;

    if (to.path === "/login") {
      // roles 仅存内存，可能是已失效会话的残留：必须由服务端确认后才弹回首页，
      // 否则失效会话的用户会被永远挡在登录页外（cookie 为 HttpOnly，无自助清会话手段）
      if (!hasRoles) {
        next();
      } else {
        try {
          await userStore.getUserInfo();
          next({ path: "/" });
        } catch {
          userStore.resetSession();
          next();
        }
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
          userStore.resetSession();
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
        userStore.resetSession();
        next(`/login?redirect=${to.path}`);
        NProgress.done();
      }
    }
  });

  router.afterEach(() => {
    NProgress.done();
  });
}
