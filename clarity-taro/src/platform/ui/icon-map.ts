/**
 * 图标名与 Vant iconfont 码点的对应关系。
 *
 * 两端共用同一份字体（vant-icon.ttf），所以视觉完全一致：
 * - 小程序 / H5 走 Taroify 的图标组件（见 Icon.tsx）
 * - RN 直接把码点渲染成字体字符（见 Icon.rn.tsx）
 *
 * 新增图标时从 @taroify/icons/index.css 的 `.van-icon-xxx:before{content:"..."}`
 * 取码点补进来，同时确认 Taroify 有同名组件（PascalCase 文件）。
 */
export const ICON_UNICODE: Record<string, string> = {
  "arrow-left": "\ue668",
  arrow: "\ue660",
  "arrow-up": "\ue65f",
  "arrow-down": "\ue65e",
  edit: "\ue6a8",
  delete: "\ue6a4",
  add: "\ue65d",
  plus: "\ue6fa",
  star: "\ue727",
  "star-o": "\ue722",
  "photo-o": "\ue700",
  photo: "\ue708",
  search: "\ue710",
  photograph: "\ue719",
  lock: "\ue6dc",
  gift: "\ue6c6",
  fire: "\ue6b6",
  cross: "\ue6a7",
  close: "\ue690",
  success: "\ue728",
  "setting-o": "\ue745",
  phone: "\ue6ff",
  "phone-o": "\ue6fc",
  expand: "\ue6b4",
  "expand-o": "\ue6b2",
  "delete-o": "\ue6a6",
  "bulb-o": "\ue67b",
  "bar-chart-o": "\ue671",
  "brush-o": "\ue67f",
  "calendar-o": "\ue67e",
  "chart-trending-o": "\ue688",
  clock: "\ue68e",
  "clock-o": "\ue68f",
  description: "\ue643",
  "description-o": "\ue6a3",
  exchange: "\ue6af",
  "good-job-o": "\ue6c5",
  "manager-o": "\ue6e3",
  "question-o": "\ue70b",
  "warning-o": "\ue743",
  "flag-o": "\ue6ba",
  "filter-o": "\ue6b5",
};

/** iconfont 在 RN 里的字体族名，与 vant-icon.ttf 内部的 name 表一致 */
export const ICON_FONT_FAMILY = "vant-icon";

export type IconName = keyof typeof ICON_UNICODE;
