import { SESSION_KEY } from "dehaze-sdk-js";

/** 页面路径（单一来源，供各页面与跳转复用） */
export const LOGIN_PATH = "/pages/login/index";
export const HOME_PATH = "/pages/home/index";

/** 免登录白名单页面（含首页品牌页） */
const WHITE_LIST = [
  "pages/login/index",
  "pages/register/index",
  "pages/home/index",
];

function isWhitePath(path: string): boolean {
  const normalized = path.replace(/^\//, "");
  return WHITE_LIST.some((item) => normalized.startsWith(item));
}

function hasValidSession(): boolean {
  try {
    return !!uni.getStorageSync(SESSION_KEY);
  } catch {
    return false;
  }
}

const INTERCEPT_METHODS = [
  "navigateTo",
  "redirectTo",
  "reLaunch",
  "switchTab",
] as const;

function authInterceptor(args: { url: string }): boolean {
  const path = args.url.split("?")[0] || "";
  if (!isWhitePath(path) && !hasValidSession()) {
    uni.reLaunch({ url: LOGIN_PATH });
    return false;
  }
  return true;
}

/** 安装路由守卫：拦截未登录访问非白名单页面 */
export function setupRouteGuard() {
  INTERCEPT_METHODS.forEach((method) =>
    uni.addInterceptor(method, { invoke: authInterceptor })
  );
}

/**
 * 检查初始登录态（应用首启自动加载的首页不经过拦截器，需显式检查）
 * 无登录态且当前页非白名单时跳转登录页
 */
export function checkInitialAuth() {
  if (hasValidSession()) return;
  const pages = getCurrentPages();
  const current = pages.length > 0 ? pages[pages.length - 1]!.route || "" : "";
  if (current && isWhitePath(current)) return;
  uni.reLaunch({ url: LOGIN_PATH });
}
