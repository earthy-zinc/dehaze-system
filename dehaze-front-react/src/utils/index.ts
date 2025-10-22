import { CssStyleObject } from "./types";

export function hasClass(el: HTMLElement, className: string) {
  const reg = new RegExp(`(^|\\s)${className}(\\s|$)`);
  return reg.test(el.className);
}

export function addClass(el: HTMLElement, className: string) {
  if (hasClass(el, className)) return;

  const newClass = el.className.split(/\s+/);
  newClass.push(className);
  el.className = newClass.join(" ");
}

export function removeClass(el: HTMLElement, className: string) {
  if (hasClass(el, className)) {
    const newClass = el.className
      .split(/\s+/)
      .filter((name) => name !== className);
    el.className = newClass.join(" ");
  }
}

const elementStyle = document.createElement("div").style as CssStyleObject;

const vendor = (() => {
  const transformNames: Record<string, string> = {
    standard: "transform",
    webkit: "webkitTransform",
    Moz: "MozTransform",
    O: "OTransform",
    ms: "msTransform",
  };

  for (const key in transformNames) {
    const val = transformNames[key];
    if (elementStyle[val] !== undefined) return key;
  }

  return false;
})();

export function prefixStyle(style: string) {
  if (vendor === false) return false;

  if (vendor === "standard") return style;

  return vendor + style.charAt(0).toUpperCase() + style.substr(1);
}

export const inBrowser = typeof window !== "undefined" && window !== null;

export const hasIntersectionObserver = checkIntersectionObserver();
const isEnumerable = Object.prototype.propertyIsEnumerable;
const getSymbols = Object.getOwnPropertySymbols;

/**
 * 取值
 * @param {Object | Array} form
 * @param  {...any} selectors
 * @returns
 */
export function getValue(form: any, ...selectors: string[]) {
  const res = selectors.map((s) => {
    return s
      .replace(/\[(\w+)\]/g, ".$1")
      .split(".")
      .reduce((prev, cur) => {
        return prev && prev[cur];
      }, form);
  });
  return res;
}

/**
 * 防抖
 * @param {*} fn
 * @param {*} delay
 * @returns
 */
export function debounce(fn: (args?: any) => void, delay: number) {
  let timer: NodeJS.Timeout | null;
  return function (this: any, ...args: any) {
    if (timer) {
      clearTimeout(timer);
    }
    timer = null;
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

/**
 * 判断是否支持IntersectionObserver
 * @returns {boolean}
 */
export function checkIntersectionObserver(): boolean {
  if (
    inBrowser &&
    "IntersectionObserver" in window &&
    "IntersectionObserverEntry" in window &&
    "intersectionRatio" in window.IntersectionObserverEntry.prototype
  ) {
    // Minimal polyfill for Edge 15's lack of `isIntersecting`
    // See: https://github.com/w3c/IntersectionObserver/issues/211
    if (!("isIntersecting" in window.IntersectionObserverEntry.prototype)) {
      Object.defineProperty(
        window.IntersectionObserverEntry.prototype,
        "isIntersecting",
        {
          get() {
            return this.intersectionRatio > 0;
          },
        }
      );
    }
    return true;
  }
  return false;
}

/**
 * is object
 *
 * @param {*} val
 * @returns {boolean}
 */
export function isObject(val: any): boolean {
  return typeof val === "function" || toString.call(val) === "[object Object]";
}

/**
 * is primitive
 *
 * @param {*} val
 * @returns {boolean}
 */
export function isPrimitive(val: any): boolean {
  return typeof val === "object" ? val === null : typeof val !== "function";
}

/**
 * check private key
 *
 * @export
 * @param {*} key
 * @returns {boolean}
 */
export function isValidKey(key: any): boolean {
  return key !== "__proto__" && key !== "constructor" && key !== "prototype";
}

/**
 * Assign the enumerable es6 Symbol properties from one
 * or more objects to the first object passed on the arguments.
 * Can be used as a supplement to other extend, assign or
 * merge methods as a polyfill for the Symbols part of
 * the es6 Object.assign method.
 * https://github.com/jonschlinkert/assign-symbols
 *
 * @param {*} target
 * @param {Array} args
 * @returns
 */
function assignSymbols(target: any, ...args: any[]) {
  if (!isObject(target))
    throw new TypeError("expected the first argument to be an object");

  if (
    args.length === 0 ||
    typeof Symbol !== "function" ||
    typeof getSymbols !== "function"
  )
    return target;

  for (const arg of args) {
    const names = getSymbols(arg);

    for (const key of names) {
      if (isEnumerable.call(arg, key)) target[key] = arg[key];
    }
  }
  return target;
}

/**
 * Deeply assign the values of all enumerable-own-properties and symbols
 * from one or more source objects to a target object. Returns the target object.
 * https://github.com/jonschlinkert/assign-deep
 *
 * @param {*} target
 * @param {Array} args
 * @returns
 */
export function assign(target: any, ...args: any[]): void {
  let i = 0;
  if (isPrimitive(target)) target = args[i++];
  if (!target) target = {};
  for (; i < args.length; i++) {
    if (isObject(args[i])) {
      for (const key of Object.keys(args[i])) {
        if (isValidKey(key)) {
          if (isObject(target[key]) && isObject(args[i][key]))
            assign(target[key], args[i][key]);
          else target[key] = args[i][key];
        }
      }
      assignSymbols(target, args[i]);
    }
  }
  return target;
}

/**
 * 判断是否是外部链接
 *
 * @param {string} path
 * @returns {Boolean}
 */
export function isExternal(path: string) {
  const isExternal = /^(https?:|http?:|mailto:|tel:)/.test(path);
  return isExternal;
}

export function hexToRGBA(hex: string, alpha: number) {
  const r = parseInt(hex.slice(1, 3), 16),
    g = parseInt(hex.slice(3, 5), 16),
    b = parseInt(hex.slice(5, 7), 16);

  if (alpha) {
    return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
  } else {
    return "rgb(" + r + ", " + g + ", " + b + ")";
  }
}

export function loadImage(
  url: string,
  crossOrigin: boolean
): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      resolve(image);
    };
    image.onerror = (err) => {
      reject(new Error("Image load error"));
    };
    if (crossOrigin) image.crossOrigin = "Anonymous"; // 支持跨域图片

    image.src = url;
  });
}

const host = window.location.host + import.meta.env.VITE_JAVA_BASE_API;

export function changeUrl(url: string) {
  if (!url) return "";
  const oldHost = new URL(url).host;
  return url.replace(oldHost, host);
}
