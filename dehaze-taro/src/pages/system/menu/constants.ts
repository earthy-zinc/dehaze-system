import { MenuTypeEnum } from "dehaze-sdk-js";
import type { MenuForm } from "dehaze-sdk-js";

// 菜单类型配置
export const MENU_TYPE_CONFIG: Record<
  string,
  { label: string; color: "primary" | "success" | "warning" | "info" }
> = {
  [MenuTypeEnum.CATALOG]: { label: "目录", color: "primary" },
  [MenuTypeEnum.MENU]: { label: "菜单", color: "success" },
  [MenuTypeEnum.BUTTON]: { label: "按钮", color: "warning" },
  [MenuTypeEnum.EXTLINK]: { label: "外链", color: "info" },
};

// 菜单类型选项
export const MENU_TYPE_OPTIONS = [
  { value: MenuTypeEnum.CATALOG, label: "目录" },
  { value: MenuTypeEnum.MENU, label: "菜单" },
  { value: MenuTypeEnum.BUTTON, label: "按钮" },
  { value: MenuTypeEnum.EXTLINK, label: "外链" },
];

// 默认表单
export const DEFAULT_FORM: MenuForm = {
  type: MenuTypeEnum.CATALOG,
  parentId: 0,
  name: "",
  path: "",
  component: "",
  perm: "",
  icon: "",
  redirect: "",
  visible: 1,
  sort: 1,
};
