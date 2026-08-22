import { MenuForm, MenuQuery, MenuVO } from "@/api/menu/model";
import { MenuTypeEnum } from "@/enums/MenuTypeEnum";
import { uniqueName, uniqueCode } from "./common";
import { TestCleanupRegistry } from "#/utils/cleanup";
import MenuAPI from "@/api/menu";

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

/**
 * 在菜单树中按名称递归查找菜单 ID（内联实现，工厂文件不含测试辅助函数）
 * @param menus 菜单树
 * @param name 目标名称
 */
function findMenuIdByName(menus: MenuVO[], name: string): number | null {
  for (const m of menus) {
    if (m.name === name) return m.id ?? null;
    if (m.children && m.children.length > 0) {
      const found = findMenuIdByName(m.children, name);
      if (found != null) return found;
    }
  }
  return null;
}

/**
 * 创建多级菜单链，返回各级菜单 ID（按层级升序）
 * 注意：MenuAPI.add 返回 void，需要通过名称搜索获取 ID
 * @param rootParentId 根父菜单 ID（链顶挂载在哪个菜单下，0 表示根级）
 * @param levels 要创建的层级数
 * @param cleanup 清理注册表（可选）
 * @returns 各级菜单 ID 数组，索引 0 为第一级新建菜单
 */
export async function createMenuChain(
  rootParentId: number,
  levels: number,
  cleanup?: TestCleanupRegistry
): Promise<number[]> {
  const ids: number[] = [];

  let currentParentId = rootParentId;
  for (let i = 0; i < levels; i++) {
    // MENU 类型带 component，可正常创建各级菜单
    const form = createMenuForm({ parentId: currentParentId, type: MenuTypeEnum.MENU });
    const menuName = form.name!;
    await MenuAPI.add(form);

    // 【后端接口限制】MenuAPI.add 返回 void，需通过名称搜索获取 ID
    const menuList = await MenuAPI.getList(createMenuQuery({ keywords: menuName }));
    const menuId = findMenuIdByName(menuList, menuName);
    if (menuId == null) {
      throw new Error(`创建菜单链失败：未找到名为 "${menuName}" 的菜单`);
    }
    ids.push(menuId);

    if (cleanup) {
      cleanup.register(async () => {
        try {
          await MenuAPI.deleteByIds(String(menuId));
        } catch {
          /* 忽略 */
        }
      });
    }
    currentParentId = menuId;
  }

  return ids;
}
