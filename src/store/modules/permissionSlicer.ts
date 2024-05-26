import { RouteVO } from "@/api/menu/model";

function hasPermission(roles: string[], route: RouteVO) {
  if (route.meta && route.meta.roles) {
    // 角色【超级管理员】拥有所有权限，忽略校验
    if (roles.includes("ROOT")) {
      return true;
    }
    return roles.some((role) => route.meta?.roles?.includes(role));
  } else {
    return true;
  }
}

function filterAsyncRouter(routes: RouteVO[], roles: string[]) {
  const asyncRoutes: RouteVO[] = [];
  routes.forEach((route) => {
    const tmpRoute = { ...route };
    if (hasPermission(roles, tmpRoute)) {
      if (tmpRoute.component?.toString() === "Layout") {
        // TODO
      } else {
        const URL = `/src/pages/${tmpRoute.component}.tsx`;
        import(URL).then((module) => {
          tmpRoute.component = module.default;
        });
      }

      if (tmpRoute.children && tmpRoute.children.length > 0) {
        filterAsyncRouter(tmpRoute.children, roles);
      }

      asyncRoutes.push(tmpRoute);
    }
  });

  return asyncRoutes;
}
