/**
 * 运行环境探测。
 *
 * RN 的全局 window 是 globalThis 别名，但无 DOM API（location/addEventListener/PerformanceObserver），
 * 不能用 typeof window 区分浏览器与 RN；RN 的可靠标识是 navigator.product === "ReactNative"。
 *
 * 导出函数而非常量：运行时才求值，测试 stub 全局对象后判断才生效。
 */

/** 真实浏览器环境：window 存在且具备 DOM 事件 API */
export function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof window.addEventListener === "function";
}

/** React Native 环境（Hermes/JSC） */
export function isRN(): boolean {
  return (
    typeof navigator !== "undefined" &&
    (navigator as { product?: string }).product === "ReactNative"
  );
}
