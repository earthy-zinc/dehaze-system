/**
 * 路由守卫
 *
 * uni-app 没有 Vue Router，路由守卫通过在 App.vue 中
 * 使用 uni.addInterceptor 拦截导航 API 实现。
 *
 * 使用方式：在 App.vue onLaunch 中调用 setupRouteGuard()
 */

import { ACCESS_TOKEN_KEY } from "@/api/config";

/** 白名单页面（无需登录即可访问） */
const WHITE_LIST = ["pages/login/index"];

/** 登录页路径 */
const LOGIN_PATH = "pages/login/index";

/** 首页路径 */
const HOME_PATH = "pages/home/index";

/** 检查是否需要登录 */
function isWhitePath(path: string): boolean {
  // 兼容带 / 或不带 / 前缀的路径
  const normalized = path.replace(/^\//, "");
  return WHITE_LIST.some((item) => normalized.startsWith(item));
}

/** 获取当前页面路径（不带前导 /） */
function getCurrentPagePath(): string {
  const pages = getCurrentPages();
  if (pages.length > 0) {
    return pages[pages.length - 1]!.route || "";
  }
  return "";
}

/** 检查是否有有效 Token */
function hasValidToken(): boolean {
  try {
    const token = uni.getStorageSync(ACCESS_TOKEN_KEY);
    return !!token;
  } catch {
    return false;
  }
}

/** 安装路由守卫 */
export function setupRouteGuard() {
  // 拦截 navigateTo
  uni.addInterceptor("navigateTo", {
    invoke(args) {
      const path = (args as { url: string }).url.split("?")[0] || "";
      if (!isWhitePath(path) && !hasValidToken()) {
        uni.reLaunch({ url: `/${LOGIN_PATH}` });
        return false;
      }
      return true;
    },
  });

  // 拦截 redirectTo
  uni.addInterceptor("redirectTo", {
    invoke(args) {
      const path = (args as { url: string }).url.split("?")[0] || "";
      if (!isWhitePath(path) && !hasValidToken()) {
        uni.reLaunch({ url: `/${LOGIN_PATH}` });
        return false;
      }
      return true;
    },
  });

  // 拦截 reLaunch
  uni.addInterceptor("reLaunch", {
    invoke(args) {
      const path = (args as { url: string }).url.split("?")[0] || "";
      if (!isWhitePath(path) && !hasValidToken()) {
        uni.reLaunch({ url: `/${LOGIN_PATH}` });
        return false;
      }
      return true;
    },
  });

  // 拦截 switchTab
  uni.addInterceptor("switchTab", {
    invoke(args) {
      const path = (args as { url: string }).url.split("?")[0] || "";
      if (!isWhitePath(path) && !hasValidToken()) {
        uni.reLaunch({ url: `/${LOGIN_PATH}` });
        return false;
      }
      return true;
    },
  });
}

/**
 * 启动时检查登录态
 *
 * uni-app 的 addInterceptor 不会拦截应用首次启动时自动加载的首页，
 * 因此需要在 App.vue onLaunch 中显式调用此方法：
 * 若未登录且当前不在白名单页面，则跳转到登录页。
 */
export function checkInitialAuth() {
  if (hasValidToken()) return;
  const current = getCurrentPagePath();
  if (current && isWhitePath(current)) return;
  // 当前页面需要登录但无 token，跳转登录页
  uni.reLaunch({ url: `/${LOGIN_PATH}` });
}

/** 跳转到登录页 */
export function navigateToLogin() {
  uni.reLaunch({ url: `/${LOGIN_PATH}` });
}

/** 跳转到首页 */
export function navigateToHome() {
  uni.reLaunch({ url: `/${HOME_PATH}` });
}
