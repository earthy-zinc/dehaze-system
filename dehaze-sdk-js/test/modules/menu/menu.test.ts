import { MenuAPI, MenuForm, MenuVO, RouteVO } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createMenuForm, createMenuQuery, createMenuChain } from "#/factories/menu";
import { uniqueCode } from "#/factories/common";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { MenuTypeEnum } from "@/enums/MenuTypeEnum";
import { USERS } from "#/factories/constants";
import { login } from "#/utils/auth";

/** 在菜单树中按条件递归查找菜单 */
function findMenu(menus: MenuVO[], predicate: (menu: MenuVO) => boolean): MenuVO | null {
  for (const menu of menus) {
    if (predicate(menu)) return menu;
    if (menu.children && menu.children.length > 0) {
      const found = findMenu(menu.children, predicate);
      if (found) return found;
    }
  }
  return null;
}

/** 在菜单树中按名称递归查找菜单 */
function findMenuByName(menus: MenuVO[], name: string): MenuVO | null {
  return findMenu(menus, (menu) => menu.name === name);
}

/** 递归收集路由树中的全部 path */
function collectRoutePaths(routes: RouteVO[]): string[] {
  const paths: string[] = [];
  const traverse = (list: RouteVO[]) => {
    list.forEach((r) => {
      if (typeof r.path === "string" && r.path.length > 0) paths.push(r.path);
      if (r.children && r.children.length > 0) traverse(r.children);
    });
  };
  traverse(routes);
  return paths;
}

/** 在路由树中按 path 递归查找路由 */
function findRouteByPath(routes: RouteVO[], path: string): RouteVO | null {
  for (const r of routes) {
    if (r.path === path) return r;
    if (r.children && r.children.length > 0) {
      const found = findRouteByPath(r.children, path);
      if (found) return found;
    }
  }
  return null;
}

describe("菜单管理接口测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdMenuIds: number[] = [];

  afterAll(async () => {
    await cleanup.executeAll();
  });

  // 按创建顺序反向清理，保证先删子菜单再删父菜单
  cleanup.registerIds(
    () => createdMenuIds,
    (id) => MenuAPI.deleteByIds(id)
  );

  /** 创建菜单并按名称回查 ID（MenuAPI.add 返回 void，不返回新菜单 ID） */
  async function createMenuAndGetId(form: MenuForm, track = true): Promise<number> {
    await MenuAPI.add(form);
    const list = await MenuAPI.getList(createMenuQuery({ keywords: form.name! }));
    const createdMenu = findMenuByName(list, form.name!);
    expect(createdMenu).not.toBeNull();
    const id = createdMenu!.id!;
    if (track) createdMenuIds.push(id);
    return id;
  }

  describe("GET /api/v1/menus/routes - 获取路由列表", () => {
    test("正向测试：获取路由列表并验证路由结构", async () => {
      const result = await MenuAPI.getRoutes();

      expect(Array.isArray(result)).toBe(true);

      // 系统必有带 path 的菜单路由，保证下方结构断言必然执行（避免条件化空转）
      const wellFormed = result.filter((r) => typeof r.path === "string" && r.path.length > 0);
      expect(wellFormed.length).toBeGreaterThan(0);
      const route = wellFormed[0]!;
      expect(typeof route.name).toBe("string");
      expect(route.name!.length).toBeGreaterThan(0);
    });
  });

  describe("验证：路由列表过滤", () => {
    test("验证：路由列表不包含按钮类型菜单", async () => {
      // 自建按钮类型菜单（type=4），验证其 path 不会进入路由列表
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const buttonForm = createMenuForm({
        parentId,
        type: MenuTypeEnum.BUTTON,
        perm: uniqueCode("test:perm"),
      });
      await createMenuAndGetId(buttonForm);
      const buttonPath = buttonForm.path!;

      const routes = await MenuAPI.getRoutes();

      // 递归收集路由中的全部 path
      const collectPaths = (routeList: typeof routes, acc: string[]): string[] => {
        routeList.forEach((r) => {
          if (typeof r.path === "string") acc.push(r.path);
          if (r.children && r.children.length > 0) collectPaths(r.children, acc);
        });
        return acc;
      };
      const routePaths = collectPaths(routes, []);

      // 按钮类型菜单不应出现在路由列表中
      expect(routePaths).not.toContain(buttonPath);

      // 保留原结构校验：具备完整字段的路由应携带 name 与 meta
      const verifyRoute = (routeList: typeof routes) => {
        routeList.forEach((r) => {
          if (typeof r.path === "string" && r.path.length > 0) {
            expect(typeof r.name).toBe("string");
            expect(r.meta).toBeDefined();
          }
          if (r.children && r.children.length > 0) {
            verifyRoute(r.children);
          }
        });
      };
      verifyRoute(routes);
    });
  });

  describe("GET /api/v1/menus - 获取菜单树形列表", () => {
    test("正向测试：获取菜单树形列表并验证树形结构", async () => {
      const result = await MenuAPI.getList(createMenuQuery());

      expect(Array.isArray(result)).toBe(true);

      if (result.length > 0) {
        const menu = result[0];
        if (menu) {
          expect(menu.id).toBeGreaterThan(0);
          expect(menu.name).toBeTruthy();
          expect(menu.type).toBeTruthy();

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
      const allMenus = await MenuAPI.getList(createMenuQuery());
      expect(allMenus.length).toBeGreaterThan(0);

      const firstMenu = allMenus[0]!;
      const searchKeyword = firstMenu.name!.substring(0, 2);
      const result = await MenuAPI.getList(createMenuQuery({ keywords: searchKeyword }));

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const verifyKeyword = (menus: typeof result, keyword: string) => {
        menus.forEach((menu) => {
          if (menu.name) {
            expect(menu.name.toLowerCase().includes(keyword.toLowerCase())).toBe(true);
          }
          if (menu.children && menu.children.length > 0) {
            verifyKeyword(menu.children, keyword);
          }
        });
      };
      verifyKeyword(result, searchKeyword);
    });

    test("正向测试：按权限标识筛选菜单（命中目标且排除无关菜单）", async () => {
      // 自建两个不同 perm 的按钮菜单，验证 perm 模糊筛选真正生效（T-MM-007）
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const permA = uniqueCode("test:perm");
      const permB = uniqueCode("test:perm");
      const buttonAId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: permA })
      );
      const buttonBId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: permB })
      );

      const result = await MenuAPI.getList(createMenuQuery({ perm: permA }));

      const ids: number[] = [];
      const collectIds = (menus: MenuVO[]) => {
        menus.forEach((m) => {
          if (m.id) ids.push(m.id);
          if (m.children) collectIds(m.children);
        });
      };
      collectIds(result);

      expect(ids).toContain(buttonAId);
      expect(ids).not.toContain(buttonBId);
    });

    test("正向测试：按菜单类型筛选菜单", async () => {
      // 后端查询参数 type 为整数（2=CATALOG），响应 type 为字符串枚举名"CATALOG"
      const result = await MenuAPI.getList(createMenuQuery({ type: 2 }));
      expect(Array.isArray(result)).toBe(true);
      const verifyType = (menus: typeof result) => {
        menus.forEach((m) => {
          expect(m.type).toBe(MenuTypeEnum.CATALOG);
          if (m.children) verifyType(m.children);
        });
      };
      verifyType(result);
    });

    test("正向测试：按显示状态筛选菜单", async () => {
      const result = await MenuAPI.getList(createMenuQuery({ visible: 1 }));
      expect(Array.isArray(result)).toBe(true);
      const verifyVisible = (menus: typeof result) => {
        menus.forEach((m) => {
          expect(m.visible).toBe(1);
          if (m.children) verifyVisible(m.children);
        });
      };
      verifyVisible(result);
    });
  });

  describe("GET /api/v1/menus/options - 获取菜单下拉数据源", () => {
    test("正向测试：获取菜单下拉列表并验证数据准确性", async () => {
      const result = await MenuAPI.getOptions();

      expect(Array.isArray(result)).toBe(true);

      if (result.length > 0) {
        const firstOption = result[0]!;
        expect(typeof firstOption.value).toBe("number");
        expect(typeof firstOption.label).toBe("string");
        expect(firstOption.label!.length).toBeGreaterThan(0);
      }
    });

    test("验证：菜单下拉选项不包含按钮类型", async () => {
      // 创建按钮类型菜单后，验证其 ID 不出现在下拉选项中
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const buttonId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: uniqueCode("test:perm") })
      );

      const options = await MenuAPI.getOptions();
      const collectOptionIds = (opts: typeof options): number[] => {
        const ids: number[] = [];
        const traverse = (list: typeof options) => {
          list.forEach((o) => {
            if (typeof o.value === "number") {
              ids.push(o.value);
            }
            if (o.children) traverse(o.children);
          });
        };
        traverse(opts);
        return ids;
      };
      expect(collectOptionIds(options)).not.toContain(buttonId);
    });
  });

  describe("GET /api/v1/menus/{id}/form - 获取菜单表单数据", () => {
    let testMenuId: number;
    let testMenuName: string;

    beforeAll(async () => {
      const form = createMenuForm({ parentId: 0 });
      testMenuName = form.name!;
      testMenuId = await createMenuAndGetId(form);
    });

    test("正向测试：获取菜单表单数据并验证数据完整性", async () => {
      const result = await MenuAPI.getFormData(testMenuId);

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
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.CATALOG });
      // 【后端接口限制】POST /api/v1/menus 返回 ResultVoid，不返回创建的菜单ID
      const menuId = await createMenuAndGetId(form);

      const menuInfo = await MenuAPI.getFormData(menuId);
      expect(menuInfo.name).toBe(form.name);
      expect(menuInfo.type).toBe(form.type);
    });

    test("正向测试：创建菜单类型并验证数据", async () => {
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU });
      const menuId = await createMenuAndGetId(form);

      const menuInfo = await MenuAPI.getFormData(menuId);
      expect(menuInfo.type).toBe(MenuTypeEnum.MENU);
    });

    test("正向测试：创建按钮类型并验证数据", async () => {
      // 按钮类型必须携带权限标识 perm
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const buttonId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: uniqueCode("test:perm") })
      );

      const buttonInfo = await MenuAPI.getFormData(buttonId);
      expect(buttonInfo.type).toBe(MenuTypeEnum.BUTTON);
      expect(buttonInfo.parentId).toBe(parentId);
    });

    test("正向测试：创建子菜单并验证父子关系", async () => {
      const parentMenuId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.CATALOG })
      );
      const childMenuId = await createMenuAndGetId(
        createMenuForm({ parentId: parentMenuId, type: MenuTypeEnum.MENU })
      );

      const childInfo = await MenuAPI.getFormData(childMenuId);
      expect(childInfo.parentId).toBe(parentMenuId);

      const fullMenuList = await MenuAPI.getList(createMenuQuery());
      const createdChild = findMenu(fullMenuList, (m) => m.id === childMenuId);
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

      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：缺少必需字段 type 应抛出业务错误", async () => {
      const form: any = {
        parentId: 0,
        name: "测试菜单",
      };

      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("正向测试：创建外链类型菜单并验证数据", async () => {
      const form = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.EXTLINK,
        path: "https://www.baidu.com",
      });
      const menuId = await createMenuAndGetId(form);

      const menuInfo = await MenuAPI.getFormData(menuId);
      expect(menuInfo.type).toBe(MenuTypeEnum.EXTLINK);
    });

    test("参数校验：同一父级下菜单名称已存在应失败", async () => {
      const form = createMenuForm({ parentId: 0 });
      const menuName = form.name!;
      await createMenuAndGetId(form);

      // path 用不同值，避免触发防重复提交幂等拦截，确保真正走重名校验 A0501
      const dupForm = createMenuForm({
        parentId: 0,
        name: menuName,
        path: uniqueCode("/dup"),
      });
      await expectBizError(MenuAPI.add(dupForm), ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：权限标识已存在应失败", async () => {
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const permValue = uniqueCode("test:perm");
      await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: permValue })
      );

      // 相同 perm 的按钮应创建失败
      const dupForm = createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: permValue });
      await expectBizError(MenuAPI.add(dupForm), ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：上级菜单不能是按钮类型", async () => {
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      const buttonId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.BUTTON, perm: uniqueCode("test:perm") })
      );

      const childForm = createMenuForm({ parentId: buttonId, type: MenuTypeEnum.MENU });
      await expectBizError(MenuAPI.add(childForm), ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：菜单类型缺少路由地址应失败", async () => {
      const form = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.MENU,
        path: "",
        component: "",
      });
      await expectBizError(MenuAPI.add(form), ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：按钮类型缺少权限标识应失败", async () => {
      const form = createMenuForm({
        parentId: 0,
        type: MenuTypeEnum.BUTTON,
        perm: "",
      });
      await expectBizError(MenuAPI.add(form), ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界：在5级菜单下新增第6级应失败（超出层级限制）", async () => {
      // 根级算第 1 级，新建 5 级菜单到第 5 级，再尝试新增第 6 级
      const chainIds = await createMenuChain(0, 5, cleanup);

      const form = createMenuForm({ parentId: chainIds[4]!, type: MenuTypeEnum.MENU });
      await expectBizError(MenuAPI.add(form), [
        "A0504",
        "A0503",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/menus/{id} - 修改菜单", () => {
    let testMenuId: number;

    beforeAll(async () => {
      testMenuId = await createMenuAndGetId(createMenuForm({ parentId: 0 }));
    });

    test("正向测试：更新菜单名称并验证更新真实生效", async () => {
      const originalMenu = await MenuAPI.getFormData(testMenuId);
      const newForm = createMenuForm({ parentId: originalMenu.parentId ?? 0 });

      await MenuAPI.update(testMenuId.toString(), newForm);

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
      await expectBizError(MenuAPI.update("99999999", form), ["A0401"]);
    });

    test("参数校验：修改后菜单名称与同级菜单重复应失败", async () => {
      // 创建两个同级菜单，将 menu2 的名称改为 menu1 的名称应失败
      const form1 = createMenuForm({ parentId: 0 });
      const form2 = createMenuForm({ parentId: 0 });
      const menu1Id = await createMenuAndGetId(form1);
      const menu2Id = await createMenuAndGetId(form2);

      await expectBizError(MenuAPI.update(String(menu2Id), { ...form2, name: form1.name! }), [
        "A0501",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：上级菜单不能设置为自己", async () => {
      const originalMenu = await MenuAPI.getFormData(testMenuId);
      await expectBizError(
        MenuAPI.update(String(testMenuId), { ...originalMenu, parentId: testMenuId } as MenuForm),
        ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("参数校验：上级菜单不能设置为自己的子菜单（循环引用检测）", async () => {
      const childMenuId = await createMenuAndGetId(
        createMenuForm({ parentId: testMenuId, type: MenuTypeEnum.MENU })
      );

      const originalMenu = await MenuAPI.getFormData(testMenuId);
      await expectBizError(
        MenuAPI.update(String(testMenuId), {
          ...originalMenu,
          parentId: childMenuId,
        } as MenuForm),
        ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });
  });

  describe("PATCH /api/v1/menus/{menuId} - 修改菜单显示状态", () => {
    let testMenuId: number;

    beforeAll(async () => {
      testMenuId = await createMenuAndGetId(createMenuForm({ parentId: 0 }));
    });

    test("正向测试：更新显示状态并验证生效", async () => {
      await MenuAPI.updateVisible(testMenuId, 0);
      const menuInfo = await MenuAPI.getFormData(testMenuId);
      expect(menuInfo.visible).toBe(0);

      await MenuAPI.updateVisible(testMenuId, 1);
      const restored = await MenuAPI.getFormData(testMenuId);
      expect(restored.visible).toBe(1);
    });

    test("异常测试：更新不存在菜单应抛出业务错误", async () => {
      await expectBizError(MenuAPI.updateVisible(99999999, 0), ["A0401"]);
    });
  });

  describe("DELETE /api/v1/menus/{ids} - 删除菜单", () => {
    test("正向测试：删除单个菜单并验证菜单真的被删除", async () => {
      const menuId = await createMenuAndGetId(createMenuForm({ parentId: 0 }), false);

      await MenuAPI.deleteByIds(String(menuId));

      await expectBizError(MenuAPI.getFormData(menuId), ["A0401"]);
    });

    test("正向测试：批量删除多个菜单并验证全部被删除", async () => {
      const id1 = await createMenuAndGetId(createMenuForm({ parentId: 0 }), false);
      const id2 = await createMenuAndGetId(createMenuForm({ parentId: 0 }), false);

      await MenuAPI.deleteByIds(`${id1},${id2}`);

      await expectBizError(MenuAPI.getFormData(id1), ["A0401"]);
      await expectBizError(MenuAPI.getFormData(id2), ["A0401"]);
    });

    test("异常测试：删除不存在的菜单应抛出业务错误", async () => {
      await expectBizError(MenuAPI.deleteByIds("99999999"), ["A0401"]);
    });

    test("正向测试：删除父菜单时级联删除子菜单", async () => {
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.CATALOG }),
        false
      );
      const childId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.MENU }),
        false
      );

      await MenuAPI.deleteByIds(String(parentId));

      await expectBizError(MenuAPI.getFormData(parentId), ["A0401"]);
      await expectBizError(MenuAPI.getFormData(childId), ["A0401"]);
    });

    test("正向测试：批量删除含父子关系的菜单（去重无误报）", async () => {
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.CATALOG }),
        false
      );
      const childId = await createMenuAndGetId(
        createMenuForm({ parentId, type: MenuTypeEnum.MENU }),
        false
      );

      await MenuAPI.deleteByIds(`${parentId},${childId}`);

      await expectBizError(MenuAPI.getFormData(parentId), ["A0401"]);
      await expectBizError(MenuAPI.getFormData(childId), ["A0401"]);
    });
  });

  describe("边界测试：菜单管理", () => {
    test("超长菜单名称应被拒绝", async () => {
      const form = createMenuForm({ name: "x".repeat(500), parentId: 0 });
      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("特殊字符菜单名称不应污染存储", async () => {
      // 名称唯一生成：删除为逻辑删除且重名校验含软删行，固定名称跨运行复创必撞 A0501
      const specialName = `测试<>&"'菜单${uniqueCode("t")}`;
      const form = createMenuForm({ name: specialName, parentId: 0 });

      await MenuAPI.add(form);

      const menuList = await MenuAPI.getList(createMenuQuery({ keywords: specialName }));
      const found = findMenuByName(menuList, specialName);
      expect(found).not.toBeNull();
      if (found?.id) {
        createdMenuIds.push(found.id);
        expect(found.name).not.toMatch(/<[^>]+>/);
      }
    });
  });

  describe("参数边界与对抗性语料", () => {
    test("边界：菜单名称长度上限 64 字符应创建成功且原样存储", async () => {
      // 唯一后缀占位，避免逻辑删除软删行导致的跨运行重名 A0501
      const suffix = uniqueCode("n");
      const name64 = "m".repeat(64 - suffix.length) + suffix;
      const form = createMenuForm({ name: name64, parentId: 0 });
      const id = await createMenuAndGetId(form);
      const info = await MenuAPI.getFormData(id);
      expect(info.name).toBe(name64);
    });

    test("参数校验：超长菜单名称（65 字符）应被拒绝", async () => {
      const form = createMenuForm({ name: "m".repeat(65), parentId: 0 });
      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("对抗性脏语料名称应原样存储（全半角/CRLF/BOM/零宽/emoji/注入特征）", async () => {
      const parentId = await createMenuAndGetId(
        createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU })
      );
      // 各名称附加唯一可见后缀：MySQL utf8mb4 ai_ci 排序规则将 BOM/零宽字符视为
      // ignorable，纯零宽差异的同父名称会被重名校验判重（防伪装，合理行为）
      const dirtyNames = [
        `ＭＥＮＵ／测试${uniqueCode("t")}`, // 全角字符
        `menu\r\n测试${uniqueCode("t")}`, // CRLF
        `\uFEFFmenu测试${uniqueCode("t")}`, // BOM
        `menu\u200B测试${uniqueCode("t")}`, // 零宽空格
        `菜单🚀🧪test${uniqueCode("t")}`, // emoji
        `menu'"-- ;测试${uniqueCode("t")}`, // SQL 注入特征字符（应被参数化查询安全处理）
      ];
      for (const name of dirtyNames) {
        const form = createMenuForm({ parentId, type: MenuTypeEnum.MENU, name });
        const id = await createMenuAndGetId(form);
        const info = await MenuAPI.getFormData(id);
        expect(info.name).toBe(name);
      }
    });

    test("边界：sort=0 允许（排在同级最前，T-MM-052）", async () => {
      const form = createMenuForm({ parentId: 0, sort: 0 });
      const id = await createMenuAndGetId(form);
      const info = await MenuAPI.getFormData(id);
      expect(info.sort).toBe(0);
    });

    test("参数校验：sort 为负数应被拒绝（T-MM-051）", async () => {
      const form = createMenuForm({ parentId: 0, sort: -1 });
      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：显示状态 visible=2 应被拒绝", async () => {
      const id = await createMenuAndGetId(createMenuForm({ parentId: 0 }));
      await expectBizError(MenuAPI.updateVisible(id, 2), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("路由过滤与缓存失效链路", () => {
    test("验证：隐藏菜单路由以 meta.hidden 标记，恢复显示后解除（T-MM-059）", async () => {
      // 行为变更：隐藏菜单保留在路由列表中（前端可做灰度/隐藏展示），由 meta.hidden 标记
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU });
      const id = await createMenuAndGetId(form);

      await MenuAPI.updateVisible(id, 0);
      let hit = findRouteByPath(await MenuAPI.getRoutes(), form.path!);
      expect(hit).not.toBeNull();
      expect(hit!.meta?.hidden).toBe(true);

      await MenuAPI.updateVisible(id, 1);
      hit = findRouteByPath(await MenuAPI.getRoutes(), form.path!);
      expect(hit).not.toBeNull();
      expect(hit!.meta?.hidden).toBe(false);
    });

    test("验证：删除菜单后路由缓存失效，不再返回该路由（T-MM-042）", async () => {
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU });
      const id = await createMenuAndGetId(form);

      expect(collectRoutePaths(await MenuAPI.getRoutes())).toContain(form.path!);

      await MenuAPI.deleteByIds(String(id));
      const idx = createdMenuIds.indexOf(id);
      if (idx >= 0) createdMenuIds.splice(idx, 1);

      expect(collectRoutePaths(await MenuAPI.getRoutes())).not.toContain(form.path!);
    });

    test("回归：批量删除重复 ID 应去重成功而非误报菜单不存在（T-MM-044）", async () => {
      // 曾暴露缺陷：count_by_ids 去重计数与 len(menu_ids) 比对不一致 → 误报 A0401，已修复
      const id = await createMenuAndGetId(createMenuForm({ parentId: 0 }));
      await MenuAPI.deleteByIds(`${id},${id}`);
      await expectBizError(MenuAPI.getFormData(id), ["A0401"]);
    });

    test("回归：删除非数字 ID 应返回参数错误 A0400 而非 500", async () => {
      // 曾暴露缺陷：int() 转换未捕获 ValueError → 500，已修复为 A0400
      await expectBizError(MenuAPI.deleteByIds("abc"), ["A0400"]);
    });
  });

  describe("路由 meta.roles 角色可见性标注（T-MM-062）", () => {
    test("新建菜单路由 meta.roles 仅含默认分配的 ROOT+ADMIN，不含未分配角色", async () => {
      // 契约：/menus/routes 返回全量路由，每条路由 meta.roles 标注可见角色编码，
      // 前端按当前用户角色过滤（后端不按用户裁剪，安全性依赖 meta.roles 标注正确）
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.MENU });
      await createMenuAndGetId(form);

      const hit = findRouteByPath(await MenuAPI.getRoutes(), form.path!);
      expect(hit).not.toBeNull();
      expect(hit!.meta?.roles).toEqual(["ADMIN", "ROOT"]);
      expect(hit!.meta?.roles).not.toContain("USER");
    });
  });

  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户新增菜单应失败", async () => {
      const form = createMenuForm({ parentId: 0, type: MenuTypeEnum.CATALOG });
      // Python 后端 require_permission 抛 403，全局异常统一映射为 A0301（未授权）
      await expectBizError(MenuAPI.add(form), ["A0301"]);
    });

    test("边界：普通用户修改菜单应失败", async () => {
      // 需携带满足 MenuForm 的完整字段，使请求到达 require_permission 后返回 A0301
      const form = createMenuForm();
      await expectBizError(MenuAPI.update("1", { ...form, name: "hacked" } as MenuForm), ["A0301"]);
    });

    test("边界：普通用户删除菜单应失败", async () => {
      await expectBizError(MenuAPI.deleteByIds("1"), ["A0301"]);
    });
  });
});
