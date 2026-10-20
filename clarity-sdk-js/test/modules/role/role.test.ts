import { ImportExportAPI, MenuAPI, RoleAPI, RoleForm, RoleQuery, UserAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createRoleForm, createRoleQuery } from "#/factories/role";
import { uniqueCode, uniqueName } from "#/factories/common";
import { ROLES } from "#/factories/constants";
import { createUserForm } from "#/factories/user";
import { SEED_PASSWORD } from "#/config/constant";
import { login, forceLogin } from "#/utils/auth";

describe("角色管理接口测试", () => {
  // 统一管理创建的角色ID，用于清理
  const createdRoleIds: number[] = [];
  afterAll(async () => {
    for (const roleId of createdRoleIds) {
      try {
        await RoleAPI.deleteByIds(roleId.toString());
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    }
  });

  // 创建一个角色并返回其 id（自动登记清理）；失败即抛出，保证用例前置条件
  async function createRoleAndGetId(overrides?: Partial<RoleForm>): Promise<number> {
    const form = createRoleForm(overrides);
    await RoleAPI.add(form);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
    const createdRole = pageResult.list.find((role) => role.code === form.code);
    if (!createdRole?.id) {
      throw new Error("角色创建失败，无法找到创建的角色");
    }
    createdRoleIds.push(createdRole.id);
    return createdRole.id;
  }

  describe("GET /api/v1/roles/page - 角色分页列表", () => {
    test("获取角色分页列表并验证数据结构与业务字段", async () => {
      const query = createRoleQuery({ pageNum: 1, pageSize: 10 });
      const result = await RoleAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);

      if (result.list.length > 0) {
        const firstRole = result.list[0]!;
        expect(firstRole.id).toBeGreaterThan(0);
        expect(typeof firstRole.name).toBe("string");
        expect(typeof firstRole.code).toBe("string");
      }
    });

    test("按关键词搜索并验证搜索结果准确性", async () => {
      const allRoles = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 100 }));
      expect(allRoles.list.length).toBeGreaterThan(0);

      const sampleRole = allRoles.list[0]!;
      const searchKeyword = sampleRole.name!.substring(0, 2);

      const result = await RoleAPI.getPage(createRoleQuery({ keywords: searchKeyword }));
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeGreaterThan(0);

      result.list.forEach((item) => {
        const matchName = item.name!.toLowerCase().includes(searchKeyword.toLowerCase());
        const matchCode = item.code!.toLowerCase().includes(searchKeyword.toLowerCase());
        expect(matchName || matchCode).toBe(true);
      });
    });

    test("分页逻辑验证 - 不同页码返回不同数据", async () => {
      const pageSize = 5;
      const page1Result = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize }));
      const page2Result = await RoleAPI.getPage(createRoleQuery({ pageNum: 2, pageSize }));

      expect(page1Result.list.length).toBeLessThanOrEqual(pageSize);
      expect(page2Result.list.length).toBeLessThanOrEqual(pageSize);

      // 验证分页隔离性
      if (page1Result.list.length > 0 && page2Result.list.length > 0) {
        const page1Ids = page1Result.list.map((r) => r.id);
        const page2Ids = page2Result.list.map((r) => r.id);
        const hasIntersection = page1Ids.some((id) => page2Ids.includes(id));
        expect(hasIntersection).toBe(false);
      }
    });

    test("获取角色下拉列表并验证数据格式", async () => {
      const result = await RoleAPI.getOptions();

      expect(Array.isArray(result)).toBe(true);

      if (result.length > 0) {
        const firstOption = result[0]!;
        expect(firstOption.value).toBeGreaterThan(0);
        expect(firstOption.label).toBeTruthy();
      }
    });
  });

  describe("GET /api/v1/roles/{roleId}/menuIds - 获取角色的菜单ID集合", () => {
    test("获取角色的菜单ID集合并验证幂等性", async () => {
      const pageResult = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const roleId = pageResult.list[0]!.id!;

      const result = await RoleAPI.getRoleMenuIds(roleId);
      expect(Array.isArray(result)).toBe(true);
      if (result.length > 0) {
        result.forEach((menuId) => {
          expect(menuId).toBeGreaterThan(0);
        });
      }
    });

    test("获取不存在角色的菜单ID集合应返回空", async () => {
      // 后端对不存在的角色返回成功但 data 为空数组（Jackson 序列化为 []，SDK 解析为 []）
      // 若 data 为 null 则 SDK 解析为 undefined，两种情况均视为"无菜单"
      const result = await RoleAPI.getRoleMenuIds(99999999);
      expect(result === undefined || (Array.isArray(result) && result.length === 0)).toBe(true);
    });
  });

  describe("PATCH /api/v1/roles/{roleId}/menus - 分配菜单(包括按钮权限)给角色", () => {
    let testRoleId: number;
    let originalMenuIds: number[] = [];

    beforeAll(async () => {
      testRoleId = await createRoleAndGetId();
      originalMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
    });

    test("为角色分配菜单并验证权限确实被分配", async () => {
      const menuIds = [1, 2, 3];

      await RoleAPI.updateRoleMenus(testRoleId, menuIds);

      const currentMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      expect(Array.isArray(currentMenuIds)).toBe(true);
      menuIds.forEach((id) => {
        expect(currentMenuIds).toContain(id);
      });
    });

    test("清空角色菜单并验证权限确实被清空", async () => {
      await RoleAPI.updateRoleMenus(testRoleId, []);

      const currentMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      expect(currentMenuIds).toEqual([]);

      // 恢复原始权限
      await RoleAPI.updateRoleMenus(testRoleId, originalMenuIds);
    });

    test("更新角色菜单权限并验证权限确实被更新", async () => {
      const firstMenuIds = [10, 11, 12];
      const secondMenuIds = [13, 14, 15];

      await RoleAPI.updateRoleMenus(testRoleId, firstMenuIds);
      let currentMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      firstMenuIds.forEach((id) => {
        expect(currentMenuIds).toContain(id);
      });

      await RoleAPI.updateRoleMenus(testRoleId, secondMenuIds);
      currentMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      secondMenuIds.forEach((id) => {
        expect(currentMenuIds).toContain(id);
      });

      firstMenuIds.forEach((id) => {
        expect(currentMenuIds).not.toContain(id);
      });
    });

    test("为不存在的角色分配菜单应返回业务错误", async () => {
      await expectBizError(RoleAPI.updateRoleMenus(99999999, [1, 2, 3]), "A0401", "不存在");
    });

    test("参数校验：分配不存在的菜单应失败", async () => {
      await expectBizError(RoleAPI.updateRoleMenus(testRoleId, [99999999]), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
      // 验证原有权限未被修改
      const currentMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      expect(currentMenuIds).not.toContain(99999999);
    });

    test("验证：分配菜单权限后正确回显", async () => {
      // 先清空，再分配已知菜单 ID
      await RoleAPI.updateRoleMenus(testRoleId, []);
      const menuIdsToAssign = [1, 2, 3];
      await RoleAPI.updateRoleMenus(testRoleId, menuIdsToAssign);

      const echoedMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      expect(Array.isArray(echoedMenuIds)).toBe(true);
      menuIdsToAssign.forEach((id) => {
        expect(echoedMenuIds).toContain(id);
      });

      // 恢复原始权限
      await RoleAPI.updateRoleMenus(testRoleId, originalMenuIds);
    });
  });

  describe("GET /api/v1/roles/{roleId}/form - 角色表单数据", () => {
    test("获取角色表单数据并验证数据完整性", async () => {
      const pageResult = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const role = pageResult.list[0]!;
      const roleId = role.id!;

      const result = await RoleAPI.getFormData(roleId);

      expect(result.id).toBe(roleId);
      expect(result.name).toBe(role.name);
      expect(result.code).toBe(role.code);
      expect(result.status).toBe(role.status);
      expect(result.sort).toBe(role.sort);

      if (result.dataScope !== undefined) {
        expect(typeof result.dataScope).toBe("number");
      }
    });

    test("获取不存在角色的表单数据应返回空", async () => {
      // 后端对不存在的角色返回成功但 data 为空（Jackson 省略 null 字段，SDK 解析为 undefined）
      const result = await RoleAPI.getFormData(99999999);
      expect(result).toBeUndefined();
    });
  });

  describe("POST /api/v1/roles - 新增角色", () => {
    test("创建角色并验证数据真实持久化", async () => {
      const form = createRoleForm({
        sort: 100,
        dataScope: 2,
      });

      const result = await RoleAPI.add(form);

      // SDK 响应拦截器从 envelope 中提取 data 字段；写操作 data 为 null，Jackson 省略该字段，SDK 返回 undefined
      expect(result).toBeUndefined();

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const createdRole = pageResult.list.find((role) => role.code === form.code);
      expect(createdRole).toBeDefined();
      expect(createdRole?.name).toBe(form.name);
      expect(createdRole?.status).toBe(form.status);
      expect(createdRole?.sort).toBe(form.sort);

      if (createdRole?.id) {
        const formData = await RoleAPI.getFormData(createdRole.id);
        expect(formData.dataScope).toBe(2);
        createdRoleIds.push(createdRole.id);
      }
    });

    test("创建带默认dataScope的角色并验证", async () => {
      const form = createRoleForm({ dataScope: 1, sort: 101 });

      await RoleAPI.add(form);

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const createdRole = pageResult.list.find((role) => role.code === form.code);
      expect(createdRole).toBeDefined();
      expect(createdRole?.code).toBe(form.code);

      const formData = await RoleAPI.getFormData(createdRole!.id!);
      expect(formData.id).toBe(createdRole!.id);
      expect(formData.dataScope).toBe(1);
      createdRoleIds.push(createdRole!.id!);
    });

    test("创建禁用状态的角色并验证状态值", async () => {
      const form = createRoleForm({ status: 0, sort: 102 });

      await RoleAPI.add(form);

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const createdRole = pageResult.list.find((role) => role.code === form.code);
      expect(createdRole).toBeDefined();
      expect(createdRole?.status).toBe(0);
      if (createdRole?.id) {
        createdRoleIds.push(createdRole.id);
      }
    });

    test("正向测试：创建全部数据权限角色（dataScope=0）并验证", async () => {
      const form = createRoleForm({ dataScope: 0, sort: 103 });
      await RoleAPI.add(form);

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const createdRole = pageResult.list.find((role) => role.code === form.code);
      expect(createdRole).toBeDefined();

      const formData = await RoleAPI.getFormData(createdRole!.id!);
      expect(formData.dataScope).toBe(0);
      createdRoleIds.push(createdRole!.id!);
    });

    // T-RM-012：后端已去除 RoleForm.dataScope 的 schema 默认值（原 default=0 架空必填
    // 校验），缺省时 service 抛 A0400「数据权限不能为空」。
    test("参数校验：缺少数据权限应失败", async () => {
      const form: Partial<RoleForm> = {
        name: "测试角色",
        code: uniqueCode("TEST_ROLE"),
        status: 1,
        sort: 100,
      };
      await expectBizError(RoleAPI.add(form as RoleForm), ["A0400"], "数据权限不能为空");
    });

    test("参数校验：缺少必需字段 code", async () => {
      const form: Partial<RoleForm> = {
        name: "测试角色",
        status: 1,
      };

      await expectBizError(RoleAPI.add(form as RoleForm), ["A0400", "B0001"]);
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form: Partial<RoleForm> = {
        code: "TEST_ROLE",
        status: 1,
      };

      await expectBizError(RoleAPI.add(form as RoleForm), ["A0400", "B0001"]);
    });

    test("参数校验：角色编码已存在", async () => {
      const form = createRoleForm({ code: ROLES.ADMIN.code });
      await expectBizError(RoleAPI.add(form), "A0501", ["编码", "code"]);
    });
  });

  describe("PUT /api/v1/roles/{id} - 修改角色", () => {
    let testRoleId: number;
    let originalRole: RoleForm;

    beforeAll(async () => {
      testRoleId = await createRoleAndGetId({ dataScope: 1 });
      originalRole = await RoleAPI.getFormData(testRoleId);
    });

    test("更新角色名称并验证更新真实生效", async () => {
      const newName = `更新后的角色名称_${Date.now()}`;
      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: newName,
        status: originalRole.status!,
        dataScope: originalRole.dataScope || 1,
      };

      await RoleAPI.update(testRoleId, form as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.name).toBe(newName);
      expect(formData.code).toBe(originalRole.code);
      expect(formData.status).toBe(originalRole.status);
    });

    test("更新角色状态并验证状态值正确", async () => {
      const disableForm: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: originalRole.name,
        status: 0,
        dataScope: originalRole.dataScope || 1,
      };
      await RoleAPI.update(testRoleId, disableForm as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.status).toBe(0);

      await RoleAPI.update(testRoleId, {
        id: testRoleId,
        code: originalRole.code,
        name: originalRole.name,
        status: 1,
        dataScope: originalRole.dataScope || 1,
      } as RoleForm);

      const formData2 = await RoleAPI.getFormData(testRoleId);
      expect(formData2.status).toBe(1);
    });

    test("更新角色排序并验证排序值正确", async () => {
      const newSort = 999;
      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: originalRole.name,
        sort: newSort,
        dataScope: originalRole.dataScope || 1,
      };

      await RoleAPI.update(testRoleId, form as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.sort).toBe(newSort);
    });

    test("更新角色数据权限并验证权限值正确", async () => {
      const newDataScope = 3;
      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: originalRole.name,
        dataScope: newDataScope,
      };

      await RoleAPI.update(testRoleId, form as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.dataScope).toBe(newDataScope);
    });

    test("同时更新多个字段并验证所有字段都更新成功", async () => {
      const newName = `批量更新名称_${Date.now()}`;
      const newSort = 888;
      const newStatus = 0;
      const newDataScope = 3;

      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: newName,
        sort: newSort,
        status: newStatus,
        dataScope: newDataScope,
      };

      await RoleAPI.update(testRoleId, form as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.name).toBe(newName);
      expect(formData.sort).toBe(newSort);
      expect(formData.status).toBe(newStatus);
      expect(formData.dataScope).toBe(newDataScope);
      expect(formData.code).toBe(originalRole.code);

      // 恢复原状态
      await RoleAPI.update(testRoleId, {
        id: testRoleId,
        code: originalRole.code,
        name: originalRole.name,
        status: 1,
        sort: originalRole.sort || 0,
        dataScope: originalRole.dataScope || 1,
      } as RoleForm);
    });

    test("更新不存在的角色应返回业务错误", async () => {
      const form: Partial<RoleForm> = {
        name: "测试",
      };

      await expectBizError(RoleAPI.update(99999999, form as RoleForm), ["A0400", "B0001"]);
    });

    test("参数校验：角色编码冲突", async () => {
      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: ROLES.ADMIN.code,
        name: originalRole.name,
        dataScope: originalRole.dataScope || 1,
      };

      await expectBizError(RoleAPI.update(testRoleId, form as RoleForm), [
        "A0503",
        "B0001",
        "A0400",
      ]);

      // 验证未被修改
      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.code).toBe(originalRole.code);
    });
  });

  describe("PATCH /api/v1/roles/{roleId}/status - 角色状态管理", () => {
    let testRoleId: number;
    let originalStatus: number;

    beforeAll(async () => {
      testRoleId = await createRoleAndGetId({ status: 1 });
      originalStatus = (await RoleAPI.getFormData(testRoleId)).status!;
    });

    test("正向测试：禁用角色并验证状态变更", async () => {
      await RoleAPI.updateStatus(testRoleId, 0);
      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.status).toBe(0);
    });

    test("正向测试：启用角色并验证状态变更", async () => {
      await RoleAPI.updateStatus(testRoleId, 1);
      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.status).toBe(1);
    });

    test("异常测试：禁用超级管理员角色应失败", async () => {
      // 内置角色（ROOT）不可修改状态，后端返回 A0503（OPERATION_NOT_ALLOW）。
      // 若后端保护失效，本用例会真实禁用 ROOT 并连锁污染其他模块的权限用例，
      // 故失败路径上保底恢复 ROOT 为启用状态。
      try {
        await expectBizError(RoleAPI.updateStatus(ROLES.ROOT.id, 0), [
          "A0503",
          "A0233",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await RoleAPI.updateStatus(ROLES.ROOT.id, 1).catch(() => {});
      }
    });

    test("异常测试：更新不存在角色的状态应失败", async () => {
      await expectBizError(RoleAPI.updateStatus(99999999, 0), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/roles/{ids} - 删除角色", () => {
    let testRoleIds: number[] = [];

    beforeAll(async () => {
      for (let i = 0; i < 3; i++) {
        testRoleIds.push(await createRoleAndGetId({ sort: 100 + i }));
      }
      expect(testRoleIds.length).toBe(3);
    });

    test("删除单个角色并验证角色真的被删除", async () => {
      const roleId = testRoleIds[0]!;

      await RoleAPI.deleteByIds(roleId.toString());

      // 删除后查询应返回业务错误或空数据
      const result = await RoleAPI.getFormData(roleId).catch(() => null);
      expect(result === null || result === undefined || !result.id).toBe(true);
    });

    test("批量删除多个角色并验证所有角色都被删除", async () => {
      const ids = testRoleIds.slice(1);
      expect(ids.length).toBe(2);

      await RoleAPI.deleteByIds(ids.join(","));

      for (const roleId of ids) {
        const result = await RoleAPI.getFormData(roleId!).catch(() => null);
        expect(result === null || result === undefined || !result.id).toBe(true);
      }
    });

    test("删除不存在的角色应返回业务错误", async () => {
      await expectBizError(RoleAPI.deleteByIds("99999999"), "A0401", "不存在");
    });

    // 空 ID 列表属调用方编程错误，由 SDK 前置校验拦截（DELETE /{ids} 路由无法接收空路径参数）
    test("参数校验：空的ID列表由 SDK 前置校验拦截", async () => {
      await expect(RoleAPI.deleteByIds("")).rejects.toThrow("不能为空");
    });

    test("边界：删除超级管理员角色应失败（超级管理员保护）", async () => {
      // 后端对内置角色（ROOT/ADMIN）返回 A0503 OPERATION_NOT_ALLOW
      await expectBizError(RoleAPI.deleteByIds(ROLES.ROOT.id.toString()), "A0503", "不可删除");
    });

    test("边界：删除已关联用户的角色应失败", async () => {
      // ROLES.USER (id=5) 关联了多个预置用户（user、vip1、vip2、svip）
      // 后端返回 A0500 BUSINESS_ERROR
      await expectBizError(RoleAPI.deleteByIds(ROLES.USER.id.toString()), "A0500", "用户关联");
    });
  });

  // ===== 测试强化补充：对抗性语料 / 参数边界 / 软删唯一性 / 全量替换语义 =====

  describe("对抗性输入与参数边界", () => {
    test("边界：角色名称超过 64 字符应拒绝", async () => {
      const form = createRoleForm({ name: "超".repeat(65) });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("边界：角色编码超过 32 字符应拒绝", async () => {
      const form = createRoleForm({ code: "C".repeat(33) });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("安全：名称含 XSS 脚本标签应拒绝", async () => {
      const form = createRoleForm({ name: "<script>alert(1)</script>" });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("安全：编码含 javascript: 协议应拒绝", async () => {
      const form = createRoleForm({ code: "javascript:alert(1)" });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("边界：sort 为负数应拒绝", async () => {
      const form = createRoleForm({ sort: -1 });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("边界：status=2 非法状态值应拒绝", async () => {
      const form = createRoleForm({ status: 2 });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("边界：dataScope=99 非法数据权限应拒绝 [T-RM-044]", async () => {
      const form = createRoleForm({ dataScope: 99 });
      await expectBizError(RoleAPI.add(form), "A0400");
    });

    test("脏语料持久化不变量：全半角/emoji/零宽/BOM/引号名称端到端保真", async () => {
      // 合法脏语料（不含 HTML 标签）：固定脏字符核心 + 唯一后缀（sys_role.name 唯一索引
      // 含软删历史行，固定名称二次运行会命中 A0501），断言后端不静默改写、往返完全一致
      const dirtyName = `角色ＡＢＣ🎉\u200b零宽\uFEFF'\";--${uniqueCode("脏")}`;
      const form = createRoleForm({ name: dirtyName });
      await RoleAPI.add(form);
      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const created = pageResult.list.find((r) => r.code === form.code);
      expect(created).toBeDefined();
      createdRoleIds.push(created!.id!);

      const formData = await RoleAPI.getFormData(created!.id!);
      expect(formData.name).toBe(dirtyName);
    });

    test("脏语料查询：keywords 含 CRLF/引号/百分号/超长串应正常返回不报错", async () => {
      for (const keywords of ["%'\");--", "\r\n<script>", "x".repeat(300)]) {
        const result = await RoleAPI.getPage(createRoleQuery({ keywords }));
        expect(Array.isArray(result.list)).toBe(true);
      }
    });

    test("性能烟测：分页查询 3 次均在 500ms 内", async () => {
      for (let i = 0; i < 3; i++) {
        const start = Date.now();
        await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 10 }));
        expect(Date.now() - start).toBeLessThan(500);
      }
    });
  });

  describe("角色编码唯一性（uk 含软删行治理后语义）", () => {
    test("软删除后同编码角色可重建（deleted 参与唯一键，软删行不占活跃键位）", async () => {
      const form = createRoleForm();
      await RoleAPI.add(form);
      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const created = pageResult.list.find((r) => r.code === form.code);
      expect(created).toBeDefined();

      await RoleAPI.deleteByIds(created!.id!.toString());

      // 软删后同编码重建成功（uk 含 deleted，历史软删行不再占用键位）
      await RoleAPI.add({ ...form, name: uniqueName("重建角色") });

      // 活跃行唯一性仍受保护：同编码再建被拒（A0501）
      await expectBizError(
        RoleAPI.add({ ...form, name: uniqueName("活跃重复") }),
        "A0501",
        "已存在"
      );
    });
  });

  describe("菜单权限分配 - 全量替换与半选一致性", () => {
    test("回显与请求集合精确一致（全量替换/半选/清空）", async () => {
      const roleId = await createRoleAndGetId();

      // 全选父子节点
      await RoleAPI.updateRoleMenus(roleId, [1, 2, 3]);
      expect((await RoleAPI.getRoleMenuIds(roleId)).slice().sort()).toEqual([1, 2, 3]);

      // 半选：仅保留子集，回显精确等于请求集合（不得残留旧节点）
      await RoleAPI.updateRoleMenus(roleId, [2]);
      expect(await RoleAPI.getRoleMenuIds(roleId)).toEqual([2]);

      // 清空
      await RoleAPI.updateRoleMenus(roleId, []);
      expect(await RoleAPI.getRoleMenuIds(roleId)).toEqual([]);
    });
  });

  // ===== 越权访问控制：无 sys:role:* 权限用户调用写端点 =====

  describe("越权访问控制", () => {
    test("普通用户新增角色应 403 [T-RM-014]", async () => {
      await login("user");
      await expectBizError(RoleAPI.add(createRoleForm()), "A0301");
    });

    test("普通用户修改角色状态应 403 [T-RM-020 变体]", async () => {
      await login("user");
      await expectBizError(RoleAPI.updateStatus(ROLES.ROOT.id, 0), "A0301");
    });

    test("普通用户删除角色应 403 [T-RM-027]", async () => {
      await login("user");
      await expectBizError(RoleAPI.deleteByIds(ROLES.ROOT.id.toString()), "A0301");
    });

    afterAll(async () => {
      // 恢复 admin 会话供后续用例使用
      await login("admin");
    });
  });

  // ===== 权限缓存与登录快照生效链路：角色授权变更 → 重新登录 → 权限即时生效 =====

  describe("授权变更后权限快照生效链路", () => {
    let testRoleId: number;
    let testUserId: number;
    let testUsername: string;
    let permMenuId: number;
    const permProofRoleCodes: string[] = [];

    beforeAll(async () => {
      // 从菜单注册表定位 sys:role:add 按钮节点（type=button 携带 perm；列表为树形需递归查找）
      const menus = await MenuAPI.getList({ perm: "sys:role:add" });
      const findMenu = (
        list: Array<{ id?: number; perm?: string; children?: unknown[] }>
      ): { id?: number } | undefined => {
        for (const m of list) {
          if (m.perm === "sys:role:add") return m;
          if (m.children?.length) {
            const found = findMenu(
              m.children as Array<{ id?: number; perm?: string; children?: unknown[] }>
            );
            if (found) return found;
          }
        }
        return undefined;
      };
      const permMenu = findMenu(menus);
      expect(permMenu).toBeDefined();
      permMenuId = permMenu!.id!;

      // 创建测试角色并授予该接口权限
      testRoleId = await createRoleAndGetId();
      await RoleAPI.updateRoleMenus(testRoleId, [permMenuId]);

      // 创建挂载该角色的用户，重置密码为登录凭证
      testUsername = uniqueName("perm_user");
      const form = createUserForm({ username: testUsername, deptId: 1, roleIds: [testRoleId] });
      await UserAPI.add(form);
      const pageResult = await UserAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        keywords: testUsername,
      });
      const createdUser = pageResult.list.find((u) => u.username === testUsername);
      expect(createdUser).toBeDefined();
      testUserId = createdUser!.id!;
      await UserAPI.updatePassword(testUserId, SEED_PASSWORD);
    });

    test("授权后新登录用户拥有角色接口权限（可新增角色）", async () => {
      await login(testUsername);
      const form = createRoleForm({ code: uniqueCode("PERM_PROOF") });
      await RoleAPI.add(form); // 不抛错即拥有 sys:role:add
      permProofRoleCodes.push(form.code);
    });

    test("清空角色菜单并重新登录后权限即时回收（role:perms 缓存失效 + 快照刷新）", async () => {
      await login("admin");
      await RoleAPI.updateRoleMenus(testRoleId, []);

      await forceLogin(testUsername); // 强制走服务端新登录，快照基于最新角色菜单
      const form = createRoleForm({ code: uniqueCode("PERM_REVOKED") });
      permProofRoleCodes.push(form.code); // 后端缺陷可能使创建意外成功，登记以便清理
      await expectBizError(RoleAPI.add(form), "A0301");
    });

    afterAll(async () => {
      await login("admin");
      // 清理权限验证期间创建的角色
      for (const code of permProofRoleCodes) {
        try {
          const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: code }));
          const found = pageResult.list.find((r) => r.code === code);
          if (found?.id) {
            await RoleAPI.deleteByIds(found.id.toString());
          }
        } catch (e) {
          console.warn("清理权限验证角色失败:", e);
        }
      }
      try {
        await UserAPI.deleteByIds(testUserId.toString());
      } catch (e) {
        console.warn("清理测试用户失败:", e);
      }
    });
  });

  // ===== 角色导入（F-RM-008） =====

  describe("POST /api/v1/role/_import - 角色模块导入", () => {
    const importedCodes: string[] = [];

    function buildRoleCsv(rows: string[]): File {
      const header = "角色名称,角色编码,排序,状态(启用/禁用)";
      return new File([`${header}\n${rows.join("\n")}\n`], "role_import.csv", {
        type: "text/csv",
      });
    }

    const isImportResult = (
      v: unknown
    ): v is {
      totalRows: number;
      successCount: number;
      failureCount: number;
      errors: Array<{ row: number; message: string }>;
    } =>
      !!v &&
      typeof v === "object" &&
      "totalRows" in (v as object) &&
      "successCount" in (v as object);

    afterAll(async () => {
      await login("admin");
      for (const code of importedCodes) {
        try {
          const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: code }));
          const found = pageResult.list.find((r) => r.code === code);
          if (found?.id) {
            await RoleAPI.deleteByIds(found.id.toString());
          }
        } catch (e) {
          console.warn("清理导入角色失败:", e);
        }
      }
    });

    test("同步导入合法角色 CSV 返回 ImportResult [T-RM-049]", async () => {
      const code1 = uniqueCode("IMP_ROLE");
      const code2 = uniqueCode("IMP_ROLE");
      importedCodes.push(code1, code2);

      // 名称需唯一（sys_role.name 唯一索引含软删历史行，固定名称二次运行必撞）
      const file = buildRoleCsv([
        `${uniqueName("导入角色")},${code1},10,启用`,
        `${uniqueName("导入角色")},${code2},20,禁用`,
      ]);
      const result = await ImportExportAPI.import("role", { mode: "all" }, file);

      expect(isImportResult(result)).toBe(true);
      if (isImportResult(result)) {
        expect(result.totalRows).toBe(2);
        expect(result.successCount).toBe(2);
        expect(result.failureCount).toBe(0);
      }

      // 验证真实持久化
      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: code1 }));
      const imported = pageResult.list.find((r) => r.code === code1);
      expect(imported).toBeDefined();
    });

    test("导入的角色数据权限必须落在 0-3 合法域内", async () => {
      const code = uniqueCode("IMP_SCOPE");
      importedCodes.push(code);

      const file = buildRoleCsv([`${uniqueName("导入数据权限角色")},${code},10,启用`]);
      await ImportExportAPI.import("role", { mode: "all" }, file);

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: code }));
      const imported = pageResult.list.find((r) => r.code === code);
      expect(imported).toBeDefined();
      const formData = await RoleAPI.getFormData(imported!.id!);
      // 数据权限合法域为 0-3（全部/部门及子部门/本部门/本人）；越域值说明导入处理器写入了非法默认值
      expect([0, 1, 2, 3]).toContain(formData.dataScope);
    });

    test("partial 模式导入含内置角色编码的行失败且其余行成功 [T-RM-050/051]", async () => {
      const okCode = uniqueCode("IMP_PARTIAL");
      importedCodes.push(okCode);

      const file = buildRoleCsv([
        `${uniqueName("部分导入成功行")},${okCode},10,启用`,
        `冒充内置角色,${ROLES.ROOT.code},30,启用`,
      ]);
      const result = await ImportExportAPI.import("role", { mode: "partial" }, file);

      expect(isImportResult(result)).toBe(true);
      if (isImportResult(result)) {
        expect(result.successCount).toBe(1);
        expect(result.failureCount).toBe(1);
        expect(result.errors.some((e) => e.message.includes("内置角色编码不可导入"))).toBe(true);
      }
    });
  });
});
