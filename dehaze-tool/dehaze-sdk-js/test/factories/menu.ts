import { MenuForm, MenuQuery } from "@/api/menu/model";
import { MenuTypeEnum } from "@/enums/MenuTypeEnum";
import { uniqueName, uniqueCode } from "./common";

/**
 * 创建菜单表单数据
 * @param overrides 覆盖默认值的字段
 */
export function createMenuForm(overrides: Partial<MenuForm> = {}): MenuForm {
  const type = overrides.type || MenuTypeEnum.CATALOG;
  // MENU 类型必须有组件路径
  const defaultComponent = type === MenuTypeEnum.MENU ? "test/index" : "";
  return {
    parentId: 0,
    name: uniqueName("测试菜单"),
    type: MenuTypeEnum.CATALOG,
    path: "/" + uniqueCode("test"),
    component: defaultComponent,
    sort: 1,
    visible: 1,
    ...overrides,
  };
}

/**
 * 创建菜单查询参数
 * @param overrides 覆盖默认值的字段
 */
export function createMenuQuery(overrides: Partial<MenuQuery> = {}): MenuQuery {
  return {
    keywords: "",
    ...overrides,
  };
}
