import { MenuAPI, MenuVO } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createMenuForm, createMenuQuery } from "#/factories/menu";
import { MenuTypeEnum } from "@/enums/MenuTypeEnum";

/**
 * 在菜单树中递归查找指定名称的菜单
 */
function findMenuByName(menus: MenuVO[], name: string): MenuVO | null {
  for (const menu of menus) {
    if (menu.name === name) return menu;
    if (menu.children && menu.children.length > 0) {
      const found = findMenuByName(menu.children, name);
      if (found) return found;
    }
  }
  return null;
}

/**
 * 在菜单树中递归查找指定ID的菜单
 */
function findMenuById(menus: MenuVO[], id: number): MenuVO | null {
  for (const menu of menus) {
    if (menu.id === id) return menu;
    if (menu.children && menu.children.length > 0) {
      const found = findMenuById(menu.children, id);
      if (found) return found;
    }
  }
  return null;
}

describe("菜单管理接口测试", () => {
  const createdMenuIds: number[] = [];
  afterAll(async () => {
    // 清理测试创建的菜单（从后往前删除，先删子菜单）
    for (const menuId of createdMenuIds.reverse()) {
      try {
        await MenuAPI.deleteByIds(String(menuId));
      } catch (error) {
        console.warn(`清理菜单失败: ${menuId}`, error);
      }
    }
  });

  describe("GET /api/v1/menus/routes - 获取路由列表", () => {
    test("正向测试：获取路由列表并验证路由结构", async () => {
      const result = await MenuAPI.getRoutes();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);

      // 验证路由结构
      if (result.length > 0) {
        const route = result[0]!;
        expect(typeof route.path).toBe("string");
        expect(route.path!.length).toBeGreaterThan(0);
        expect(typeof route.name).toBe("string");
        expect(route.name!.length).toBeGreaterThan(0);
      }
    });
  });

  describe("GET /api/v1/menus - 获取菜单树形列表", () => {
    test("正向测试：获取菜单树形列表并验证树形结构", async () => {
      const result = await MenuAPI.getList(createMenuQuery());

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);

      // 验证树形结构
      if (result.length > 0) {
        const menu = result[0];
        if (menu) {
          expect(menu.id).toBeGreaterThan(0);
          expect(menu.name).toBeTruthy();
          expect(menu.type).toBeTruthy();

          // 如果有子菜单，验证父子关系
          if (menu.children && menu.children.length > 0) {
            menu.children.forEach((child) => {
              if (child.parentId !== undefined) {
                expect(child.parentId).toBe(menu.id);
              }
            });
          }
        }
      }
    });

    test("正向测试：按关键词搜索菜单并验证搜索结果", async () => {
      // 先获取所有菜单
      const allMenus = await MenuAPI.getList(createMenuQuery());
      expect(allMenus.length).toBeGreaterThan(0);

      const firstMenu = allMenus[0]!;
      expect(firstMenu?.name).toBeTruthy();
      const searchKeyword = firstMenu.name!.substring(0, 2);
      const query = createMenuQuery({ keywords: searchKeyword });
      const result = await MenuAPI.getList(query);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      // 验证搜索结果
      const verifyKeyword = (menus: typeof result, keyword: string) => {
        menus.forEach((menu) => {
          if (menu.name) {
            const nameContains = menu.name.toLowerCase().includes(keyword.toLowerCase());
            expect(nameContains).toBe(true);
          }
          if (menu.children && menu.children.length > 0) {
            verifyKeyword(menu.children, keyword);
          }
        });
      };
      verifyKeyword(result, searchKeyword);
    });
  });

  describe("GET /api/v1/menus/options - 获取菜单下拉数据源", () => {
    test("正向测试：获取菜单下拉列表并验证数据准确性", async () => {
      const result = await MenuAPI.getOptions();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);

      if (result.length > 0) {
        const firstOption = result[0]!;
        expect(typeof firstOption.value).toBe("number");
        expect(typeof firstOption.label).toBe("string");
        expect(firstOption.label!.length).toBeGreaterThan(0);
      }
    });
  });

  describe("GET /api/v1/menus/{id}/form - 获取菜单表单数据", () => {
    let testMenuId: number;
    let testMenuName: string;

    beforeAll(async () => {
      // 创建测试菜单
      const form = createMenuForm({ parentId: 0 });
      testMenuName = form.name!;
      await MenuAPI.add(form);

      // 通过名称查找刚创建的菜单获取ID
      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: testMenuName }));
      const createdMenu = findMenuByName(menuList, testMenuName);
      expect(createdMenu).not.toBeNull();
      testMenuId = createdMenu!.id!;
      createdMenuIds.push(testMenuId);
    });

    test("正向测试：获取菜单表单数据并验证数据完整性", async () => {
      const result = await MenuAPI.getFormData(testMenuId);

      expect(result).toBeDefined();
      // 【注意】MenuForm.id 类型定义为 string，但后端实际返回 number
      expect(result.id).toBe(testMenuId);
      expect(result.name).toBe(testMenuName);
      expect(result.type).toBeTruthy();
    });

    test("异常测试：获取不存在菜单应抛出业务错误", async () => {
      await expectBizError(MenuAPI.getFormData(99999999), ["A0401"]);
    });
  });

  describe("POST /api/v1/menus - 添加菜单", () => {
    test("正向测试：创建目录类型菜单并验证数据真实持久化", async () => {
      const form = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.CATALOG,
      });
      const menuName = form.name!;

      // 【后端接口限制】POST /api/v1/menus 返回 ResultVoid，不返回创建的菜单ID
      await MenuAPI.add(form);

      // 通过名称查找刚创建的菜单
      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: menuName }));
      const createdMenu = findMenuByName(menuList, menuName);

      expect(createdMenu).not.toBeNull();
      expect(createdMenu!.id).toBeGreaterThan(0);
      createdMenuIds.push(createdMenu!.id!);

      // 验证持久化
      const menuInfo = await MenuAPI.getFormData(createdMenu!.id!);
      expect(menuInfo.name).toBe(menuName);
      expect(menuInfo.type).toBe(form.type);
    });

    test("正向测试：创建菜单类型并验证数据", async () => {
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU });
      const menuName = form.name!;
      await MenuAPI.add(form);

      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: menuName }));
      const createdMenu = findMenuByName(menuList, menuName);

      expect(createdMenu).not.toBeNull();
      createdMenuIds.push(createdMenu!.id!);

      const menuInfo = await MenuAPI.getFormData(createdMenu!.id!);
      expect(menuInfo.type).toBe(MenuTypeEnum.MENU);
    });

    test("正向测试：创建按钮类型并验证数据", async () => {
      // 先创建父菜单
      const parentForm = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.MENU,
      });
      const parentMenuName = parentForm.name!;
      await MenuAPI.add(parentForm);

      const menuList1 = await MenuAPI.getList(createMenuQuery({ keywords: parentMenuName }));
      const parentMenu = findMenuByName(menuList1, parentMenuName);
      expect(parentMenu).not.toBeNull();
      const parentId = parentMenu!.id!;
      createdMenuIds.push(parentId);

      // 创建按钮
      const buttonForm = createMenuForm({
        parentId,
        type: MenuTypeEnum.BUTTON,
      });
      const buttonName = buttonForm.name!;
      await MenuAPI.add(buttonForm);

      const menuList2 = await MenuAPI.getList(createMenuQuery({ keywords: buttonName }));
      const buttonMenu = findMenuByName(menuList2, buttonName);
      expect(buttonMenu).not.toBeNull();
      const buttonId = buttonMenu!.id!;
      createdMenuIds.push(buttonId);

      const buttonInfo = await MenuAPI.getFormData(buttonId);
      expect(buttonInfo.type).toBe(MenuTypeEnum.BUTTON);
      expect(buttonInfo.parentId).toBe(parentId);
    });

    test("正向测试：创建子菜单并验证父子关系", async () => {
      // 先创建父菜单
      const parentForm = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.CATALOG,
      });
      const parentMenuName = parentForm.name!;
      await MenuAPI.add(parentForm);

      const menuList1 = await MenuAPI.getList(createMenuQuery({ keywords: parentMenuName }));
      const parentMenu = findMenuByName(menuList1, parentMenuName);
      expect(parentMenu).not.toBeNull();
      const parentMenuId = parentMenu!.id!;
      createdMenuIds.push(parentMenuId);

      // 创建子菜单
      const childForm = createMenuForm({
        parentId: parentMenuId,
        type: MenuTypeEnum.MENU,
      });
      const childMenuName = childForm.name!;
      await MenuAPI.add(childForm);

      const menuList2 = await MenuAPI.getList(createMenuQuery({ keywords: childMenuName }));
      const childMenu = findMenuByName(menuList2, childMenuName);
      expect(childMenu).not.toBeNull();
      const childMenuId = childMenu!.id!;
      createdMenuIds.push(childMenuId);

      // 验证父子关系
      const childInfo = await MenuAPI.getFormData(childMenuId);
      expect(childInfo.parentId).toBe(parentMenuId);

      // 验证树形结构中的父子关系
      const fullMenuList = await MenuAPI.getList(createMenuQuery());
      const createdChild = findMenuById(fullMenuList, childMenuId);
      expect(createdChild).not.toBeNull();
      if (createdChild?.parentId !== undefined) {
        expect(createdChild.parentId).toBe(parentMenuId);
      }
    });

    test("参数校验：缺少必需字段 name 应抛出业务错误", async () => {
      const form: any = {
        parentId: 0,
        type: MenuTypeEnum.CATALOG,
      };

      await expectBizError(
        MenuAPI.add(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("参数校验：缺少必需字段 type 应抛出业务错误", async () => {
      const form: any = {
        parentId: 0,
        name: "测试菜单",
      };

      await expectBizError(
        MenuAPI.add(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/menus/{id} - 修改菜单", () => {
    let testMenuId: number;

    beforeAll(async () => {
      // 创建测试用的菜单
      const form = createMenuForm({ parentId: 0 });
      const menuName = form.name!;
      await MenuAPI.add(form);

      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: menuName }));
      const createdMenu = findMenuByName(menuList, menuName);
      expect(createdMenu).not.toBeNull();
      testMenuId = createdMenu!.id!;
      createdMenuIds.push(testMenuId);
    });

    test("正向测试：更新菜单名称并验证更新真实生效", async () => {
      const originalMenu = await MenuAPI.getFormData(testMenuId);
      const newForm = createMenuForm({ parentId: originalMenu.parentId ?? 0 });

      await MenuAPI.update(testMenuId.toString(), {
        ...newForm,
      });

      // 验证更新后的数据
      const menuInfo = await MenuAPI.getFormData(testMenuId);
      expect(menuInfo.name).toBe(newForm.name);
    });

    test("正向测试：更新菜单可见性并验证状态值正确", async () => {
      const originalMenu = await MenuAPI.getFormData(testMenuId);
      const updateForm = { ...originalMenu, visible: 0 };

      await MenuAPI.update(testMenuId.toString(), updateForm);

      const menuInfo = await MenuAPI.getFormData(testMenuId);
      expect(menuInfo.visible).toBe(0);

      // 恢复可见性
      await MenuAPI.update(testMenuId.toString(), { ...originalMenu, visible: 1 });
      const restoredMenu = await MenuAPI.getFormData(testMenuId);
      expect(restoredMenu.visible).toBe(1);
    });

    test("异常测试：更新不存在的菜单应抛出业务错误", async () => {
      const form = createMenuForm();
      await expectBizError(MenuAPI.update("99999999", { ...form }), ["A0401"]);
    });
  });

  describe("DELETE /api/v1/menus/{ids} - 删除菜单", () => {
    test("正向测试：删除单个菜单并验证菜单真的被删除", async () => {
      // 创建测试菜单
      const form = createMenuForm({ parentId: 0 });
      const menuName = form.name!;
      await MenuAPI.add(form);

      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: menuName }));
      const createdMenu = findMenuByName(menuList, menuName);
      expect(createdMenu).not.toBeNull();
      const menuId = createdMenu!.id!;

      // 删除菜单
      await MenuAPI.deleteByIds(String(menuId));

      // 验证已删除
      await expectBizError(MenuAPI.getFormData(menuId), ["A0401"]);
    });

    test("正向测试：批量删除多个菜单并验证全部被删除", async () => {
      // 创建两个测试菜单
      const form1 = createMenuForm({ parentId: 0 });
      const form2 = createMenuForm({ parentId: 0 });
      await MenuAPI.add(form1);
      await MenuAPI.add(form2);

      const list1 = await MenuAPI.getList(createMenuQuery({ keywords: form1.name! }));
      const list2 = await MenuAPI.getList(createMenuQuery({ keywords: form2.name! }));
      const menu1 = findMenuByName(list1, form1.name!);
      const menu2 = findMenuByName(list2, form2.name!);
      expect(menu1).not.toBeNull();
      expect(menu2).not.toBeNull();
      const id1 = menu1!.id!;
      const id2 = menu2!.id!;

      // 批量删除
      await MenuAPI.deleteByIds(`${id1},${id2}`);

      // 验证均已删除
      await expectBizError(MenuAPI.getFormData(id1), ["A0401"]);
      await expectBizError(MenuAPI.getFormData(id2), ["A0401"]);
    });

    test("异常测试：删除不存在的菜单应抛出业务错误", async () => {
      await expectBizError(MenuAPI.deleteByIds("99999999"), ["A0401"]);
    });
  });

  describe("边界测试：菜单管理", () => {
    const createdMenuIds: number[] = [];

    afterAll(async () => {
      for (const menuId of createdMenuIds.reverse()) {
        try {
          await MenuAPI.deleteByIds(String(menuId));
        } catch {}
      }
    });

    test("超长菜单名称应被拒绝", async () => {
      const form = createMenuForm({ name: "x".repeat(500), parentId: 0 });
      await expectBizError(
        MenuAPI.add(form),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("特殊字符菜单名称不应污染存储", async () => {
      const specialName = "测试<>&\"'菜单";
      const form = createMenuForm({ name: specialName, parentId: 0 });

      const existingList = await MenuAPI.getList(createMenuQuery());
      for (const m of existingList) {
        if (m.name === specialName && m.id) {
          try {
            await MenuAPI.deleteByIds(String(m.id));
          } catch {}
        }
      }

      await MenuAPI.add(form);

      const menuList = await MenuAPI.getList(createMenuQuery());
      const found = menuList.find((m) => m.name === specialName);
      expect(found).toBeDefined();
      if (found?.id) {
        createdMenuIds.push(found.id);
        expect(found.name).not.toMatch(/<[^>]+>/);
      }
    });
  });
});
