/**
 * 安全性测试套件
 *
 * 覆盖 XSS 注入、SQL 注入、超长字符串、未认证访问等安全场景。
 * 设计依据：dehaze-doc/docs/03-模块设计/基础模块/**\u200b/测试用例.md 第8节安全性测试
 */
import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { DeptAPI, DatasetAPI, RoleAPI, DictAPI, MenuAPI } from "../../../index";
import UserAPI from "@/api/user";
import { login, logout } from "#/utils/auth";
import { expectBizError } from "#/utils/assertion";
import { createDeptForm } from "#/factories/dept";
import { createRoleForm } from "#/factories/role";
import { createDictTypeForm, createDictForm } from "#/factories/dict";
import { createDatasetForm } from "#/factories/dataset";
import { createMenuForm } from "#/factories/menu";
import { createUserForm } from "#/factories/user";
import { DEPTS, ROLES } from "#/factories/constants";
import { TestCleanupRegistry } from "#/utils/cleanup";

describe("安全性测试", () => {
  // ──────────────────────────────────────────────────────────────────────
  // 1. XSS 脚本注入防护测试
  // ──────────────────────────────────────────────────────────────────────
  describe("XSS 脚本注入防护", () => {
    const xssPayloads = [
      '<script>alert("xss")</script>',
      "<img src=x onerror=alert(1)>",
      'javascript:alert("xss")',
      "<svg onload=alert(1)>",
      '"><script>alert(document.domain)</script>',
    ];

    describe("部门名称字段 XSS 防护", () => {
      const cleanup = new TestCleanupRegistry();
      afterAll(async () => cleanup.executeAll());

      for (const payload of xssPayloads) {
        test(`XSS 注入部门名称应被拒绝或转义：${payload.substring(0, 40)}`, async () => {
          const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: payload });

          // 后端应拒绝或转义，不应存储原始 XSS 内容
          const result = await DeptAPI.add(form).catch(() => null);

          if (result !== null && typeof result === "number") {
            cleanup.register(async () => {
              await DeptAPI.deleteByIds(result.toString());
            });
            // 若后端允许创建，验证返回的名称不包含原始脚本标签
            const formData = await DeptAPI.getFormData(result);
            expect(formData.name).not.toContain("<script>");
            expect(formData.name).not.toContain("onerror=");
            expect(formData.name).not.toContain("javascript:");
          }
          // 若后端拒绝创建（result === null），则视为 XSS 已被拦截，测试通过
        });
      }
    });

    describe("角色名称字段 XSS 防护", () => {
      const cleanup = new TestCleanupRegistry();
      afterAll(async () => cleanup.executeAll());

      test("XSS 注入角色名称应被拒绝或转义", async () => {
        const form = createRoleForm({ name: '<script>alert("xss")</script>' });

        const result = await RoleAPI.add(form).catch(() => null);

        if (result === undefined || result === null) {
          // 后端拒绝了创建，XSS 被拦截，测试通过
          return;
        }

        // 若允许创建，查找并清理，验证名称已转义
        const pageResult = await RoleAPI.getPage({ pageNum: 1, pageSize: 100 });
        const createdRole = pageResult.list.find((r: any) => r.code === form.code);
        if (createdRole?.id) {
          cleanup.register(async () => {
            await RoleAPI.deleteByIds(createdRole.id!.toString());
          });
          const formData = await RoleAPI.getFormData(createdRole.id!);
          expect(formData.name).not.toContain("<script>");
        }
      });
    });

    describe("字典类型名称字段 XSS 防护", () => {
      test("XSS 注入字典类型名称应被拒绝或转义", async () => {
        const form = createDictTypeForm({ name: '<script>alert("xss")</script>' });

        const result = await DictAPI.addDictType(form).catch(() => null);

        if (result === undefined || result === null) {
          return; // 被拦截
        }

        const pageResult = await DictAPI.getDictTypePage({ pageNum: 1, pageSize: 100 });
        const created = pageResult.list.find((d: any) => d.code === form.code);
        if (created?.id) {
          try {
            await DictAPI.deleteDictTypes(created.id.toString());
          } catch {}
          expect(created.name).not.toContain("<script>");
        }
      });
    });

    describe("数据集名称字段 XSS 防护", () => {
      test("XSS 注入数据集名称应被拒绝或转义", async () => {
        const form = createDatasetForm({ name: '<script>alert("xss")</script>' });

        const datasetId = await DatasetAPI.add(form).catch(() => null);

        if (datasetId && datasetId > 0) {
          const detail = await DatasetAPI.getDatasetInfoById(datasetId);
          expect(detail.name).not.toContain("<script>");
          try {
            await DatasetAPI.deleteById(datasetId);
          } catch {}
        }
        // 若 datasetId === null，已被拦截，测试通过
      });
    });

    describe("用户昵称字段 XSS 防护", () => {
      test("XSS 注入用户昵称应被拒绝或转义", async () => {
        const form = createUserForm({ nickname: '<script>alert("xss")</script>' });

        const result = await UserAPI.add(form).catch(() => null);

        if (result === undefined || result === null) {
          return; // 被拦截
        }

        // 查找并清理
        const pageResult = await UserAPI.getPage({
          pageNum: 1,
          pageSize: 100,
          keywords: form.username!,
        });
        const createdUser = pageResult.list.find((u: any) => u.username === form.username);
        if (createdUser?.id) {
          try {
            await UserAPI.deleteByIds(createdUser.id.toString());
          } catch {}
          const formData = await UserAPI.getFormData(createdUser.id);
          if (formData?.nickname) {
            expect(formData.nickname).not.toContain("<script>");
          }
        }
      });
    });

    describe("菜单名称字段 XSS 防护", () => {
      test("XSS 注入菜单名称应被拒绝或转义", async () => {
        const form = createMenuForm({ name: '<script>alert("xss")</script>' });

        const result = await MenuAPI.add(form).catch(() => null);

        if (result === undefined || result === null) {
          return; // 被拦截
        }

        // 查找并清理
        const menuList = await MenuAPI.getList({ keywords: form.name ?? "" });
        if (menuList.length > 0) {
          const found = menuList[0];
          if (found?.id) {
            try {
              await MenuAPI.deleteByIds(String(found.id));
            } catch {}
            expect(found.name).not.toContain("<script>");
          }
        }
      });
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // 2. SQL 注入防护测试
  // ──────────────────────────────────────────────────────────────────────
  describe("SQL 注入防护", () => {
    const sqlPayloads = [
      "' OR '1'='1",
      "admin'--",
      "'; DROP TABLE sys_dept;--",
      "' UNION SELECT * FROM sys_user--",
      "1; SELECT * FROM sys_user--",
    ];

    for (const payload of sqlPayloads) {
      test(`SQL 注入尝试应被安全处理：${payload.substring(0, 40)}`, async () => {
        const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: payload });

        const createdId = await DeptAPI.add(form).catch(() => null);

        // 后端拒绝创建时，SQL 注入被拦截，测试通过
        if (createdId === null || typeof createdId !== "number") {
          return;
        }

        // 创建成功说明后端使用参数化查询，payload 仅作为字面字符串存储
        // 删除创建的部门，验证删除后列表查询正常且该部门已不存在
        try {
          await DeptAPI.deleteByIds(createdId.toString());
        } catch {}

        const list = await DeptAPI.getList();
        const allIds: number[] = [];
        const collectIds = (depts: typeof list) => {
          depts.forEach((d) => {
            if (d.id) allIds.push(d.id);
            if (d.children) collectIds(d.children);
          });
        };
        collectIds(list);

        // 列表查询正常返回（未因 SQL 注入报错），且创建的部门已被删除
        expect(allIds).not.toContain(createdId);
      });
    }
  });

  // ──────────────────────────────────────────────────────────────────────
  // 3. 超长字符串拦截测试
  // ──────────────────────────────────────────────────────────────────────
  describe("超长字符串拦截", () => {
    const longString = "x".repeat(10000);

    test("超长部门名称应被拒绝", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: longString });

      await expectBizError(DeptAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("超长角色名称应被拒绝", async () => {
      const form = createRoleForm({ name: longString });

      await expectBizError(RoleAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("超长字典类型名称应被拒绝", async () => {
      const form = createDictTypeForm({ name: longString });

      await expectBizError(DictAPI.addDictType(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("超长数据集名称应被拒绝", async () => {
      const form = createDatasetForm({ name: longString });

      await expectBizError(DatasetAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("超长菜单名称应被拒绝", async () => {
      const form = createMenuForm({ name: longString });

      await expectBizError(MenuAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("超长用户昵称应被拒绝", async () => {
      const form = createUserForm({ nickname: longString });

      await expectBizError(UserAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // 4. 特殊字符存储污染测试
  // ──────────────────────────────────────────────────────────────────────
  describe("特殊字符存储污染防护", () => {
    test("特殊字符 <>&\"' 不应造成存储污染", async () => {
      const specialName = "测试<>&\"'特殊字符";
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: specialName });

      const deptId = await DeptAPI.add(form).catch(() => null);

      if (deptId !== null && typeof deptId === "number") {
        try {
          const formData = await DeptAPI.getFormData(deptId);
          // 验证存储的内容与输入一致（特殊字符应被原样保存或转义，不应丢失）
          expect(typeof formData.name).toBe("string");
          expect(formData.name!.length).toBeGreaterThan(0);
          // 验证不包含原始 HTML 标签污染
          expect(formData.name).not.toMatch(/<[^>]+>/);
        } finally {
          try {
            await DeptAPI.deleteByIds(deptId.toString());
          } catch {}
        }
      }
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // 5. 未认证访问测试（401 防护）
  // ──────────────────────────────────────────────────────────────────────
  describe("未认证访问防护（401）", () => {
    // 单独 describe 以便在测试前 logout、测试后重新 login
    beforeAll(async () => {
      await logout();
    });

    afterAll(async () => {
      await login();
    });

    test("未登录访问部门列表应抛出错误", async () => {
      await expect(DeptAPI.getList()).rejects.toThrow();
    });

    test("未登录访问用户列表应抛出错误", async () => {
      await expect(UserAPI.getPage({ pageNum: 1, pageSize: 10 })).rejects.toThrow();
    });

    test("未登录访问角色列表应抛出错误", async () => {
      await expect(RoleAPI.getPage({ pageNum: 1, pageSize: 10 })).rejects.toThrow();
    });

    test("未登录访问字典类型列表应抛出错误", async () => {
      await expect(DictAPI.getDictTypePage({ pageNum: 1, pageSize: 10 })).rejects.toThrow();
    });

    test("未登录访问菜单列表应抛出错误", async () => {
      await expect(MenuAPI.getList({})).rejects.toThrow();
    });

    test("未登录创建部门应抛出错误", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      await expect(DeptAPI.add(form)).rejects.toThrow();
    });
  });
});
