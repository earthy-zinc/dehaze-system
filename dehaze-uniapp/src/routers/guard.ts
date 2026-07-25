import { SESSION_KEY } from "dehaze-sdk-js";

const WHITE_LIST = ["pages/login/index"];
const LOGIN_PATH = "pages/login/index";
const HOME_PATH = "pages/home/index";

function isWhitePath(path: string): boolean {
  const normalized = path.replace(/^\//, "");
  return WHITE_LIST.some((item) => normalized.startsWith(item));
}

function getCurrentPagePath(): string {
  const pages = getCurrentPages();
  if (pages.length > 0) {
    return pages[pages.length - 1]!.route || "";
  }
  return "";
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
    uni.reLaunch({ url: `/${LOGIN_PATH}` });
    return false;
  }
  return true;
}

export function setupRouteGuard() {
  INTERCEPT_METHODS.forEach((method) =>
    uni.addInterceptor(method, { invoke: authInterceptor })
  );
}

export function checkInitialAuth() {
  if (hasValidSession()) return;
  const current = getCurrentPagePath();
  if (current && isWhitePath(current)) return;
  uni.reLaunch({ url: `/${LOGIN_PATH}` });
}

export function navigateToLogin() {
  uni.reLaunch({ url: `/${LOGIN_PATH}` });
}

export function navigateToHome() {
  uni.reLaunch({ url: `/${HOME_PATH}` });
}
