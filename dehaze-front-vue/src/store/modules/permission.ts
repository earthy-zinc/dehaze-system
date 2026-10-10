import { constantRoutes } from "@/router";
import { store } from "@/store";
import { MenuAPI, RouteVO } from "dehaze-sdk-js";
import { resolveRoutePath } from "@/utils/index";
import { RouteRecordRaw } from "vue-router";

const modules = import.meta.glob("../../views/**/**.vue");
const Layout = () => import("@/layout/index.vue");

/**
 * 判断用户是否拥有路由权限
 *
 * 规则：
 * - 超级管理员（ROOT）拥有所有权限
 * - 路由未配置 meta.roles 时，所有已登录用户均可访问
 * - 路由配置了 meta.roles 时，用户角色须在列表中
 */
const hasPermission = (roles: string[], route: RouteRecordRaw) => {
  if (roles.includes("ROOT")) {
    return true;
  }
  if (route.meta && route.meta.roles && route.meta.roles.length > 0) {
    return roles.some((role) => route.meta!.roles!.includes(role));
  }
  return true;
};

/**
 * 根据组件路径或路由路径生成 PascalCase 唯一路由名称
 * 用于 keep-alive 的 include 匹配（必须与 SFC 的 name 选项一致）
 *
 * 规则：取组件路径（Layout 类型取路由路径），去除 "index" 段，连字符转驼峰，各段拼接为 PascalCase
 * 例："system/user/index" → "SystemUser"，"image-input/index" → "ImageInput"
 */
function generateRouteName(routePath: string, componentPath?: string): string {
  const source =
    !componentPath || componentPath === "Layout" ? routePath : componentPath;
  const segments = source
    .replace(/^\//, "")
    .split("/")
    .filter((s) => s && s !== "index");
  return segments
    .map((segment) =>
      segment
        .split("-")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join("")
    )
    .join("");
}

/**
 * 本次生成过程中已被占用的路由名。
 *
 * vue-router 的 addRoute 遇到**重名**会移除先前注册的同名路由：表现为该路径直接
 * 404（匹配不到任何记录）、按 name 跳转会跳到另一个模块的页面。
 * 菜单里多个模块都有 `path=index` 的子路由，其组件名派生后与父级同名而被保留为
 * 接口名 "Index"，正是同名冲突的来源。
 */
const usedRouteNames = new Set<string>();

/**
 * 由完整路径派生全局唯一的路由名（`/chat/index` → `ChatIndex`）。
 * 只在名称缺失/与父级同名/与其他路由重名时使用，避免影响既有按名跳转与 keep-alive 缓存名。
 */
function deriveUniqueRouteName(fullPath: string): string {
  const base =
    fullPath
      .split("/")
      .filter(Boolean)
      .map((segment) =>
        segment
          .replace(/[^a-zA-Z0-9]+/g, " ")
          .trim()
          .split(" ")
          .filter(Boolean)
          .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
          .join("")
      )
      .join("") || "Route";
  let name = base;
  let suffix = 2;
  while (usedRouteNames.has(name)) {
    name = `${base}${suffix++}`;
  }
  return name;
}

/**
 * 递归过滤有权限的动态路由
 *
 * @param routes 接口返回所有的动态路由
 * @param roles 用户角色集合
 * @param parentPath 父级完整路径（用于递归拼接子路由的完整路径）
 * @param parentName 父级路由名称（用于避免子路由名称与父级冲突）
 * @returns 返回用户有权限的动态路由
 */
const filterAsyncRoutes = (
  routes: RouteVO[],
  roles: string[],
  parentPath = "",
  parentName = ""
) => {
  const asyncRoutes: RouteRecordRaw[] = [];
  routes.forEach((route) => {
    const tmpRoute = { ...route } as RouteRecordRaw; // 深拷贝 route 对象 避免污染
    if (hasPermission(roles, tmpRoute)) {
      const fullPath = resolveRoutePath(parentPath, tmpRoute.path);
      // 生成 PascalCase 路由名称，用于 keep-alive include 匹配（须与 SFC name 一致）
      const routeName = generateRouteName(
        tmpRoute.path,
        tmpRoute.component?.toString()
      );
      // 名称必须全局唯一：与父级同名会造成嵌套冲突，与其他路由同名会被 addRoute 顶掉
      if (
        routeName &&
        routeName !== parentName &&
        !usedRouteNames.has(routeName)
      ) {
        tmpRoute.name = routeName;
      } else {
        tmpRoute.name = deriveUniqueRouteName(fullPath);
      }
      usedRouteNames.add(tmpRoute.name as string);
      // 子级递归时以实际生效的名称为“父级名”，避免与真实注册结果不一致
      const effectiveName = tmpRoute.name as string;
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

      if (route.children) {
        tmpRoute.children = filterAsyncRoutes(
          route.children,
          roles,
          fullPath,
          effectiveName
        );
      }

      asyncRoutes.push(tmpRoute);
    }
  });

  return asyncRoutes;
};

// setup
export const usePermissionStore = defineStore("permission", () => {
  const routes = ref<RouteRecordRaw[]>([]);

  function setRoutes(newRoutes: RouteRecordRaw[]) {
    routes.value = constantRoutes.concat(newRoutes);
  }

  /** 生成动态路由 */
  async function generateRoutes(roles: string[]) {
    const data = await MenuAPI.getRoutes();
    // 每次重建都清空名称占用表，否则重注册时会把上次的名字当成冲突而改名
    usedRouteNames.clear();
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
