import { constantRoutes } from "@/router";
import { store } from "@/store";
import { MenuAPI, RouteVO } from "dehaze-sdk-js";
import { RouteRecordRaw } from "vue-router";

const modules = import.meta.glob("../../views/**/**.vue");
const Layout = () => import("@/layout/index.vue");

/**
 * Use meta.role to determine if the current user has permission
 *
 * @param roles 用户角色集合
 * @param route 路由
 * @returns
 */
const hasPermission = (roles: string[], route: RouteRecordRaw) => {
  if (route.meta && route.meta.roles) {
    // 角色【超级管理员】拥有所有权限，忽略校验
    if (roles.includes("ROOT")) {
      return true;
    }
    return roles.some((role) => {
      if (route.meta?.roles) {
        return route.meta.roles.includes(role);
      }
    });
  }
  return false;
};

/**
 * 递归过滤有权限的动态路由
 *
 * @param routes 接口返回所有的动态路由
 * @param roles 用户角色集合
 * @param parentPath 父级完整路径（用于生成唯一 name，避免 Vue Router 4 的 name 冲突）
 * @returns 返回用户有权限的动态路由
 */
const filterAsyncRoutes = (
  routes: RouteVO[],
  roles: string[],
  parentPath = ""
) => {
  const asyncRoutes: RouteRecordRaw[] = [];
  routes.forEach((route) => {
    const tmpRoute = { ...route } as RouteRecordRaw; // 深拷贝 route 对象 避免污染
    if (hasPermission(roles, tmpRoute)) {
      // 生成基于完整路径的唯一 name，避免 Vue Router 4 在多个子路由同名时只能注册第一个的问题
      // 例如：/algorithm/list 与 /dataset/list 在原后端逻辑下 name 都为 "List"
      const fullPath = resolveFullPath(parentPath, tmpRoute.path);
      if (fullPath) {
        tmpRoute.name = fullPath;
      }
      // 如果是顶级目录，替换为 Layout 组件
      if (tmpRoute.component?.toString() == "Layout") {
        tmpRoute.component = Layout;
      } else {
        // 如果是子目录，动态加载组件
        const component = modules[`../../views/${tmpRoute.component}.vue`];
        if (component) {
          tmpRoute.component = component;
        } else {
          tmpRoute.component = modules[`../../views/error-page/404.vue`];
        }
      }

      if (tmpRoute.children) {
        tmpRoute.children = filterAsyncRoutes(
          route.children,
          roles,
          fullPath
        );
      }

      asyncRoutes.push(tmpRoute);
    }
  });

  return asyncRoutes;
};

/**
 * 拼接父子路径生成完整路径（用于路由 name 唯一化）
 * 例如：parentPath="/algorithm"，childPath="list" → "/algorithm/list"
 */
function resolveFullPath(parentPath: string, childPath: string): string {
  if (!childPath) return parentPath;
  // 子路径以 "/" 开头时视为绝对路径
  if (childPath.startsWith("/")) return childPath;
  if (!parentPath) return childPath;
  return `${parentPath.replace(/\/$/, "")}/${childPath}`;
}
// setup
export const usePermissionStore = defineStore("permission", () => {
  const routes = ref<RouteRecordRaw[]>([]);

  function setRoutes(newRoutes: RouteRecordRaw[]) {
    routes.value = constantRoutes.concat(newRoutes);
  }

  /** 生成动态路由 */
  async function generateRoutes(roles: string[]) {
    const data = await MenuAPI.getRoutes();
    const accessedRoutes = filterAsyncRoutes(data, roles);
    setRoutes(accessedRoutes);
    return accessedRoutes;
  }

  /** 获取与激活的顶部菜单项相关的混合模式左侧菜单集合 */
  const mixLeftMenus = ref<RouteRecordRaw[]>([]);
  function setMixLeftMenus(topMenuPath: string) {
    const matchedItem = routes.value.find((item) => item.path === topMenuPath);
    if (matchedItem && matchedItem.children) {
      mixLeftMenus.value = matchedItem.children;
    }
  }
  return {
    routes,
    generateRoutes,
    mixLeftMenus,
    setMixLeftMenus,
  };
});

// 非setup
export function usePermissionStoreHook() {
  return usePermissionStore(store);
}
