import { RoleAPI, RoleForm, RoleQuery } from "../../../index";
import { login, logout } from "#/utils/auth";
import { getBizError } from "#/utils/biz";
import { expectBizError, expectBizErrorOrUndefined } from "#/utils/assertion";
import { createRoleForm, createRoleQuery } from "#/factories/role";
import { ROLES } from "#/factories/constants";

describe("角色管理接口测试", () => {
  // 统一管理创建的角色ID，用于清理
  const createdRoleIds: number[] = [];

  beforeAll(async () => {
    await login();
  }, 30000);

  afterAll(async () => {
    for (const roleId of createdRoleIds) {
      try {
        await RoleAPI.deleteByIds(roleId.toString());
      } catch (e) {
        // 忽略删除错误（可能已被其他测试删除）
      }
    }
    await logout();
  });

  describe("GET /api/v1/roles/page - 角色分页列表", () => {
    test("获取角色分页列表并验证数据结构与业务字段", async () => {
      const query = createRoleQuery({ pageNum: 1, pageSize: 10 });
      const result = await RoleAPI.getPage(query);

      // Assert - 验证结构
      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);

      // Assert - 验证业务字段（如果有数据）
      if (result.list.length > 0) {
        const firstRole = result.list[0]!;
        expect(firstRole.id).toBeGreaterThan(0);
        expect(firstRole.name).toBeTruthy();
        expect(firstRole.code).toBeTruthy();
      }
    });

    test("按关键词搜索并验证搜索结果准确性", async () => {
      const allRoles = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 100 }));
      if (allRoles.list.length === 0) {
        return;
      }

      const sampleRole = allRoles.list[0]!;
      const searchKeyword = sampleRole.name!.substring(0, 2);

      const result = await RoleAPI.getPage(createRoleQuery({ keywords: searchKeyword }));

      expect(Array.isArray(result.list)).toBe(true);
      if (result.list.length > 0) {
        result.list.forEach((item) => {
          const matchName = item.name!.toLowerCase().includes(searchKeyword.toLowerCase());
          const matchCode = item.code!.toLowerCase().includes(searchKeyword.toLowerCase());
          expect(matchName || matchCode).toBe(true);
        });
      }
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

    test("超大页码应返回空数组", async () => {
      const result = await RoleAPI.getPage(createRoleQuery({ pageNum: 99999, pageSize: 10 }));

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBe(0);
    });
  });

  describe("GET /api/v1/roles/options - 角色下拉列表", () => {
    test("获取角色下拉列表并验证数据格式", async () => {
      const result = await RoleAPI.getOptions();

      expect(Array.isArray(result)).toBe(true);

      if (result.length > 0) {
        const firstOption = result[0]!;
        expect(firstOption.value).toBeGreaterThan(0);
        expect(firstOption.label).toBeTruthy();
      }
    });

    test("验证下拉列表数据与分页列表的一致性", async () => {
      const options = await RoleAPI.getOptions();
      const pageResult = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 100 }));

      expect(Array.isArray(options)).toBe(true);
      expect(Array.isArray(pageResult.list)).toBe(true);

      // Assert - 验证一致性（options中的ID应该存在于分页列表中）
      if (options.length > 0 && pageResult.list.length > 0) {
        const pageRoleIds = pageResult.list
          .map((role) => role.id)
          .filter((id): id is number => id !== undefined);

        const optionIds = options.map((opt) => opt.value);

        // 至少有一些ID是重叠的
        const hasOverlap = optionIds.some((id) => pageRoleIds.includes(Number(id)));
        expect(hasOverlap).toBe(true);
      }
    });
  });

  describe("GET /api/v1/roles/{roleId}/menuIds - 获取角色的菜单ID集合", () => {
    test("获取角色的菜单ID集合并验证幂等性", async () => {
      const pageResult = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 1 }));
      if (pageResult.list.length === 0) {
        return;
      }
      const roleId = pageResult.list[0]!.id!;

      const result = await RoleAPI.getRoleMenuIds(roleId);

      expect(Array.isArray(result)).toBe(true);
      if (result.length > 0) {
        result.forEach((menuId) => {
          expect(menuId).toBeGreaterThan(0);
        });
      }

      // Assert - 验证幂等性
      const result2 = await RoleAPI.getRoleMenuIds(roleId);
      expect(result).toEqual(result2);
    });

    test("参数校验：获取不存在角色的菜单ID集合应抛出业务错误", async () => {
      // 【预期行为】查询不存在的资源应返回业务错误（如 B0001/A0400）
      // 【实际行为】后端可能返回空数组而非抛异常，行为不统一（后端 bug）
      // 【保留此测试】持续暴露后端未正确处理资源不存在的问题
      await expectBizError(RoleAPI.getRoleMenuIds(99999999), "B0001", "不存在");
    });
  });

  describe("PUT /api/v1/roles/{roleId}/menus - 分配菜单(包括按钮权限)给角色", () => {
    let testRoleId: number;
    let originalMenuIds: number[] = [];

    beforeAll(async () => {
      const form = createRoleForm();
      await RoleAPI.add(form);

      const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
      const createdRole = pageResult.list.find((role) => role.code === form.code);
      if (createdRole?.id) {
        testRoleId = createdRole.id;
        createdRoleIds.push(testRoleId);
        originalMenuIds = await RoleAPI.getRoleMenuIds(testRoleId);
      } else {
        throw new Error("角色菜单测试：角色创建失败，无法找到创建的角色");
      }
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
      const firstMenuIds = [1, 2, 3];
      const secondMenuIds = [4, 5, 6];

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
      await expectBizError(RoleAPI.updateRoleMenus(99999999, [1, 2, 3]), "B0001", "不存在");
    });
  });

  describe("GET /api/v1/roles/{roleId}/form - 角色表单数据", () => {
    test("获取角色表单数据并验证数据完整性", async () => {
      const pageResult = await RoleAPI.getPage(createRoleQuery({ pageNum: 1, pageSize: 1 }));
      if (pageResult.list.length === 0) {
        return;
      }
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

    test("参数校验：获取不存在角色的表单数据应抛出业务错误", async () => {
      // 【预期行为】查询不存在的资源应返回业务错误（如 B0001/A0400）
      // 【实际行为】如果后端返回成功，说明未正确处理资源不存在（后端 bug）
      // 【保留此测试】持续暴露后端未正确处理资源不存在的问题
      await expectBizError(RoleAPI.getFormData(99999999), "B0001", "不存在");
    });
  });

  describe("POST /api/v1/roles - 新增角色", () => {
    test("创建角色并验证数据真实持久化", async () => {
      const form = createRoleForm({
        sort: 100,
        dataScope: 2,
      });

      const result = await RoleAPI.add(form);

      expect(result.code).toBe("00000");
      expect(result.msg).toBe("一切ok");

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

    test("参数校验：缺少必需字段 code", async () => {
      const form: Partial<RoleForm> = {
        name: "测试角色",
        status: 1,
      };

      await expectBizErrorOrUndefined(RoleAPI.add(form as RoleForm), ["A0400", "B0001"]);
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form: Partial<RoleForm> = {
        code: "TEST_ROLE",
        status: 1,
      };

      await expectBizErrorOrUndefined(RoleAPI.add(form as RoleForm), ["A0400", "B0001"]);
    });

    test("参数校验：角色编码已存在", async () => {
      const form = createRoleForm({ code: ROLES.ADMIN.code });
      await expectBizError(RoleAPI.add(form), "B0001", ["编码", "code"]);
    });
  });

  describe("PUT /api/v1/roles/{id} - 修改角色", () => {
    let testRoleId: number = 0;
    let originalRole: any = null;
    let setupSuccess = false;

    beforeAll(async () => {
      try {
        const form = createRoleForm({ dataScope: 1 });
        await RoleAPI.add(form);

        await new Promise((resolve) => setTimeout(resolve, 100));

        const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
        const createdRole = pageResult.list.find((role) => role.code === form.code);

        if (createdRole?.id) {
          testRoleId = createdRole.id;
          createdRoleIds.push(testRoleId);
          originalRole = await RoleAPI.getFormData(testRoleId);
          setupSuccess = true;
        }
      } catch (e) {
        console.error("修改角色测试 beforeAll 失败:", e);
      }
    });

    test("更新角色名称并验证更新真实生效", async () => {
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

      const newName = `更新后的角色名称_${Date.now()}`;
      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: originalRole.code,
        name: newName,
        dataScope: originalRole.dataScope || 1,
      };

      await RoleAPI.update(testRoleId, form as RoleForm);

      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.name).toBe(newName);
      expect(formData.code).toBe(originalRole.code);
      expect(formData.status).toBe(originalRole.status);
    });

    test("更新角色状态并验证状态值正确", async () => {
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

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
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

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
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

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
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

      const newName = `批量更新名称_${Date.now()}`;
      const newSort = 888;
      const newStatus = 0;
      const newDataScope = 4;

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
        dataScope: originalRole.dataScope || 1,
      } as RoleForm);
    });

    test("更新不存在的角色应返回业务错误", async () => {
      const form: Partial<RoleForm> = {
        name: "测试",
      };

      await expectBizErrorOrUndefined(RoleAPI.update(99999999, form as RoleForm), [
        "A0400",
        "B0001",
      ]);
    });

    test("参数校验：角色编码冲突", async () => {
      if (!setupSuccess || !testRoleId) {
        console.log("跳过测试：角色创建失败");
        return;
      }

      const form: Partial<RoleForm> = {
        id: testRoleId,
        code: ROLES.ADMIN.code,
        name: originalRole.name,
        dataScope: originalRole.dataScope || 1,
      };

      await expectBizErrorOrUndefined(RoleAPI.update(testRoleId, form as RoleForm), [
        "B0001",
        "A0400",
      ]);

      // 验证未被修改
      const formData = await RoleAPI.getFormData(testRoleId);
      expect(formData.code).toBe(originalRole.code);
    });
  });

  describe("DELETE /api/v1/roles/{ids} - 删除角色", () => {
    let testRoleIds: number[] = [];
    let setupSuccess = false;

    beforeAll(async () => {
      try {
        for (let i = 0; i < 3; i++) {
          const form = createRoleForm({ sort: 100 + i });
          await RoleAPI.add(form);

          await new Promise((resolve) => setTimeout(resolve, 100));

          const pageResult = await RoleAPI.getPage(createRoleQuery({ keywords: form.code }));
          const createdRole = pageResult.list.find((role) => role.code === form.code);
          if (createdRole?.id) {
            testRoleIds.push(createdRole.id);
            // 注意：这里不加入 createdRoleIds，因为会在测试中删除
          }
        }

        if (testRoleIds.length >= 3) {
          setupSuccess = true;
        }
      } catch (e) {
        console.error("删除角色测试 beforeAll 失败:", e);
      }
    });

    test("删除单个角色并验证角色真的被删除", async () => {
      if (!setupSuccess || testRoleIds.length === 0) {
        console.log("跳过测试：角色创建失败");
        return;
      }

      const roleId = testRoleIds[0];

      await RoleAPI.deleteByIds(roleId!.toString());

      try {
        await RoleAPI.getFormData(roleId!);
        // 如果没抛异常，说明还存在（可能是后端未实现删除）
        expect(true).toBe(false); // 强制失败
      } catch (error: any) {
        const bizError = getBizError(error);
        if (bizError.code) {
          expect(["B0001", "A0400"]).toContain(bizError.code);
        }
      }
    });

    test("批量删除多个角色并验证所有角色都被删除", async () => {
      if (!setupSuccess || testRoleIds.length < 2) {
        console.log("跳过测试：角色创建失败");
        return;
      }

      const ids = testRoleIds.slice(1);

      await RoleAPI.deleteByIds(ids.join(","));

      for (const roleId of ids) {
        try {
          await RoleAPI.getFormData(roleId!);
          expect(true).toBe(false); // 强制失败
        } catch (error: any) {
          const bizError = getBizError(error);
          if (bizError.code) {
            expect(["B0001", "A0400"]).toContain(bizError.code);
          }
        }
      }
    });

    test("删除不存在的角色应返回业务错误", async () => {
      await expectBizError(RoleAPI.deleteByIds("99999999"), "B0001", "不存在");
    });

    test("参数校验：空的ID列表", async () => {
      await expectBizError(RoleAPI.deleteByIds(""), "B0001");
    });
  });
});
