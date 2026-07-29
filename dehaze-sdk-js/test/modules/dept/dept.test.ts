import { DeptAPI, DeptForm, DeptQuery } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createDeptForm, createDeptQuery } from "#/factories/dept";
import { uniqueName } from "#/factories/common";
import { DEPTS } from "#/factories/constants";

describe("部门管理接口测试", () => {
  describe("GET /api/v1/dept - 部门树形表格", () => {
    test("正向测试：获取部门树形列表并验证树形结构", async () => {
      const result = await DeptAPI.getList();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      // 验证返回的数据结构完整性和父子关系
      const verifyDept = (depts: typeof result) => {
        depts.forEach((dept) => {
          expect(dept.id).toBeGreaterThan(0);
          expect(dept.name).toBeTruthy();
          expect(dept.parentId).toBeDefined();
          expect(dept.parentId).toBeGreaterThanOrEqual(0);

          // 验证子部门的 parentId 等于父部门的 id
          if (dept.children && dept.children.length > 0) {
            dept.children.forEach((child) => {
              expect(child.parentId).toBe(dept.id);
            });
            verifyDept(dept.children);
          }
        });
      };
      verifyDept(result);

      // 验证预置部门存在
      const findDeptById = (depts: typeof result, id: number): boolean => {
        for (const dept of depts) {
          if (dept.id === id) return true;
          if (dept.children && findDeptById(dept.children, id)) return true;
        }
        return false;
      };
      expect(findDeptById(result, DEPTS.CQUPT.id)).toBe(true);
    });

    test("正向测试：按关键词搜索部门并验证搜索结果", async () => {
      // 使用已知部门名称进行搜索
      const searchKeyword = DEPTS.CQUPT.name.substring(0, 2);
      const query = createDeptQuery({ keywords: searchKeyword });
      const result = await DeptAPI.getList(query);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      // 递归验证所有搜索结果都包含关键词
      const verifyKeyword = (depts: typeof result, keyword: string) => {
        depts.forEach((dept) => {
          const nameContains = dept.name!.toLowerCase().includes(keyword.toLowerCase());
          expect(nameContains).toBe(true);
          if (dept.children && dept.children.length > 0) {
            verifyKeyword(dept.children, keyword);
          }
        });
      };
      verifyKeyword(result, searchKeyword);
    });

    test("正向测试：按状态筛选部门并验证筛选结果", async () => {
      const query = createDeptQuery({ status: 1 });
      const result = await DeptAPI.getList(query);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);

      // 递归验证所有部门状态都是 1
      const verifyStatus = (depts: typeof result, status: number) => {
        depts.forEach((dept) => {
          expect(dept.status).toBe(status);
          if (dept.children && dept.children.length > 0) {
            verifyStatus(dept.children, status);
          }
        });
      };
      verifyStatus(result, 1);
    });

    test("正向测试：验证根部门的 parentId 为 0", async () => {
      const result = await DeptAPI.getList();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const rootDepts = result.filter((dept) => dept.parentId === 0);
      expect(rootDepts.length).toBeGreaterThan(0);

      rootDepts.forEach((dept) => {
        expect(dept.parentId).toBe(0);
        expect(dept.id).toBeGreaterThan(0);
        expect(dept.name).toBeTruthy();
      });

      // 验证预置根部门存在
      const cquptDept = rootDepts.find((d) => d.id === DEPTS.CQUPT.id);
      expect(cquptDept).toBeDefined();
      expect(cquptDept!.name).toBe(DEPTS.CQUPT.name);
    });
  });

  describe("GET /api/v1/dept/options - 部门下拉列表", () => {
    test("正向测试：获取部门下拉列表并验证数据准确性", async () => {
      const result = await DeptAPI.getOptions();

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      result.forEach((option) => {
        expect(option.value).toBeGreaterThan(0);
        expect(option.label).toBeTruthy();
      });

      // 验证预置部门在下拉列表中
      const optionIds = result.map((opt) => opt.value);
      expect(optionIds).toContain(DEPTS.CQUPT.id);
    });

    test("正向测试：验证下拉列表只包含启用部门", async () => {
      const options = await DeptAPI.getOptions();
      const treeResult = await DeptAPI.getList();

      expect(options.length).toBeGreaterThan(0);
      expect(treeResult.length).toBeGreaterThan(0);

      // 获取所有启用部门的 ID
      const getEnabledDeptIds = (depts: typeof treeResult): number[] => {
        const ids: number[] = [];
        const traverse = (deptList: typeof treeResult) => {
          deptList.forEach((dept) => {
            if (dept.status === 1) {
              ids.push(dept.id!);
            }
            if (dept.children && dept.children.length > 0) {
              traverse(dept.children);
            }
          });
        };
        traverse(depts);
        return ids;
      };

      const enabledDeptIds = getEnabledDeptIds(treeResult);
      const optionIds = options.map((opt) => opt.value);

      // 验证下拉列表中的部门 ID 都在启用部门列表中
      optionIds.forEach((id) => {
        expect(enabledDeptIds).toContain(id);
      });
    });
  });

  describe("GET /api/v1/dept/{deptId}/form - 获取部门表单数据", () => {
    test("正向测试：获取部门表单数据并验证数据完整性", async () => {
      const deptId = DEPTS.CQUPT.id;

      const result = await DeptAPI.getFormData(deptId);

      expect(result.id).toBe(deptId);
      expect(result.name).toBe(DEPTS.CQUPT.name);
      expect(result.parentId).toBe(DEPTS.CQUPT.parentId);
      expect([0, 1]).toContain(result.status);
    });

    test("正向测试：获取子部门的表单数据", async () => {
      const deptId = DEPTS.SOFTWARE.id;

      const result = await DeptAPI.getFormData(deptId);

      expect(result.id).toBe(deptId);
      expect(result.name).toBe(DEPTS.SOFTWARE.name);
      expect(result.parentId).toBe(DEPTS.SOFTWARE.parentId);
    });

    test("获取不存在部门的表单数据应返回空", async () => {
      // 后端对不存在的部门返回成功但 data 为空（Jackson 省略 null 字段，SDK 解析为 undefined）
      const result = await DeptAPI.getFormData(99999999);
      expect(result).toBeUndefined();
    });
  });

  describe("POST /api/v1/dept - 新增部门", () => {
    const createdDeptIds: number[] = [];

    afterAll(async () => {
      // 清理测试创建的数据，从后往前删除（先删子部门再删父部门）
      for (const deptId of createdDeptIds.reverse()) {
        try {
          await DeptAPI.deleteByIds(deptId.toString());
        } catch {
          // 忽略删除错误（可能已被级联删除）
        }
      }
    });

    test("正向测试：创建部门并验证数据真实持久化", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });

      const deptId = await DeptAPI.add(form);

      expect(deptId).toBeGreaterThan(0);
      createdDeptIds.push(deptId as number);

      // 验证持久化
      const formData = await DeptAPI.getFormData(deptId as number);
      expect(formData.id).toBe(deptId);
      expect(formData.name).toBe(form.name);
      expect(formData.parentId).toBe(form.parentId);
      expect(formData.sort).toBe(form.sort);
      expect(formData.status).toBe(form.status);
    });

    test("正向测试：创建子部门并验证父子关系", async () => {
      // 先创建父部门
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const parentDeptId = (await DeptAPI.add(parentForm)) as number;
      createdDeptIds.push(parentDeptId);

      // 再创建子部门
      const childForm = createDeptForm({ parentId: parentDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;
      createdDeptIds.push(childDeptId);

      // 验证父子关系
      const childFormData = await DeptAPI.getFormData(childDeptId);
      expect(childFormData.parentId).toBe(parentDeptId);

      // 验证树形结构中的父子关系
      const deptList = await DeptAPI.getList();
      const findDeptInTree = (depts: typeof deptList, id: number): any => {
        for (const dept of depts) {
          if (dept.id === id) return dept;
          if (dept.children) {
            const found = findDeptInTree(dept.children, id);
            if (found) return found;
          }
        }
        return null;
      };

      const createdChild = findDeptInTree(deptList, childDeptId);
      expect(createdChild).not.toBeNull();
      expect(createdChild!.parentId).toBe(parentDeptId);
    });

    test("正向测试：创建启用状态的部门并验证状态值", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, status: 1 });

      const deptId = (await DeptAPI.add(form)) as number;
      createdDeptIds.push(deptId);

      const formData = await DeptAPI.getFormData(deptId);
      expect(formData.status).toBe(1);
    });

    test("正向测试：创建禁用状态的部门并验证状态值", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, status: 0 });

      const deptId = (await DeptAPI.add(form)) as number;
      createdDeptIds.push(deptId);

      const formData = await DeptAPI.getFormData(deptId);
      expect(formData.status).toBe(0);
    });

    test("参数校验：缺少必需字段 name 应抛出业务错误", async () => {
      // 【预期行为】缺少必填字段 name 应返回参数校验错误（如 A0400/B0001）
      // 【实际行为】后端未校验 name 必填，仍创建成功（后端 bug）
      // 【保留此测试】持续暴露后端缺少必填字段校验的问题
      const form: Partial<DeptForm> = {
        parentId: DEPTS.CQUPT.id,
        status: 1,
        sort: 100,
      };

      await expectBizError(DeptAPI.add(form as DeptForm), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：缺少必需字段 parentId", async () => {
      const form: Partial<DeptForm> = {
        name: "测试部门",
        status: 1,
      };

      await expectBizError(DeptAPI.add(form as DeptForm), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：同级部门名称已存在", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });

      // 创建第一个部门
      const deptId = (await DeptAPI.add(form)) as number;
      createdDeptIds.push(deptId);

      // 尝试创建同名部门
      await expectBizError(DeptAPI.add({ ...form, sort: form.sort! + 1 }), [
        "A0501",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：父部门不存在应抛出业务错误", async () => {
      // 【预期行为】父部门 ID 不存在应返回参数校验错误（如 A0400/B0001）
      // 【实际行为】后端未校验父部门是否存在，仍创建成功（后端 bug）
      // 【保留此测试】持续暴露后端缺少父部门存在性校验的问题
      const form = createDeptForm({ parentId: 99999999 });

      await expectBizError(DeptAPI.add(form), ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("PUT /api/v1/dept/{id} - 修改部门", () => {
    let testDeptId: number;
    let originalDept: DeptForm;
    const additionalDeptIds: number[] = [];

    beforeAll(async () => {
      // 创建测试用的部门
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      testDeptId = (await DeptAPI.add(form)) as number;
      originalDept = await DeptAPI.getFormData(testDeptId);
    });

    afterAll(async () => {
      // 清理测试数据
      const allIds = [testDeptId, ...additionalDeptIds];
      for (const deptId of allIds.reverse()) {
        try {
          await DeptAPI.deleteByIds(deptId.toString());
        } catch {
          // 忽略删除错误
        }
      }
    });

    test("正向测试：更新部门名称并验证更新真实生效", async () => {
      const newForm = createDeptForm({ parentId: originalDept.parentId });

      await DeptAPI.update(testDeptId, newForm);

      // 验证更新后的数据
      const formData = await DeptAPI.getFormData(testDeptId);
      expect(formData.name).toBe(newForm.name);
      expect(formData.parentId).toBe(originalDept.parentId);

      // 恢复原名称
      await DeptAPI.update(testDeptId, { ...originalDept });
    });

    test("正向测试：更新部门状态并验证状态值正确", async () => {
      const form: Partial<DeptForm> = {
        name: originalDept.name ?? "",
        status: 0,
        parentId: originalDept.parentId,
      };

      await DeptAPI.update(testDeptId, form as DeptForm);

      const formData = await DeptAPI.getFormData(testDeptId);
      expect(formData.status).toBe(0);

      // 恢复状态
      await DeptAPI.update(testDeptId, {
        name: originalDept.name ?? "",
        status: 1,
        parentId: originalDept.parentId,
      } as DeptForm);
      const formData2 = await DeptAPI.getFormData(testDeptId);
      expect(formData2.status).toBe(1);
    });

    test("正向测试：更新部门排序并验证排序值正确", async () => {
      const newSort = 999;
      const form: Partial<DeptForm> = {
        name: originalDept.name ?? "",
        sort: newSort,
        parentId: originalDept.parentId,
      };

      await DeptAPI.update(testDeptId, form as DeptForm);

      const formData = await DeptAPI.getFormData(testDeptId);
      expect(formData.sort).toBe(newSort);
    });

    test("正向测试：移动部门到新的父部门并验证移动成功", async () => {
      // 创建新父部门
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const newParentId = (await DeptAPI.add(parentForm)) as number;
      additionalDeptIds.push(newParentId);

      // 移动部门
      const form: Partial<DeptForm> = {
        name: originalDept.name ?? "",
        parentId: newParentId,
      };
      await DeptAPI.update(testDeptId, form as DeptForm);

      // 验证移动
      const formData = await DeptAPI.getFormData(testDeptId);
      expect(formData.parentId).toBe(newParentId);

      // 恢复原父部门
      await DeptAPI.update(testDeptId, {
        name: originalDept.name ?? "",
        parentId: originalDept.parentId,
      } as DeptForm);
    });

    test("异常测试：更新不存在的部门", async () => {
      const form = createDeptForm();

      await expectBizError(DeptAPI.update(99999999, form), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：部门名称冲突（同级）", async () => {
      // 创建另一个部门
      const anotherForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const anotherDeptId = (await DeptAPI.add(anotherForm)) as number;
      additionalDeptIds.push(anotherDeptId);

      // 尝试将 testDept 改为与 anotherDept 同名
      await expectBizError(
        DeptAPI.update(testDeptId, {
          name: anotherForm.name!,
          parentId: DEPTS.CQUPT.id,
        } as DeptForm),
        ["A0501", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("参数校验：不能将部门设置为自己的子部门应抛出业务错误", async () => {
      // 创建子部门
      const childForm = createDeptForm({ parentId: testDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;
      additionalDeptIds.push(childDeptId);

      // 【预期行为】将部门移动到其子部门下应返回业务错误（如 A0400/B0001），防止循环依赖
      // 【实际行为】后端未校验循环依赖，更新成功（后端 bug）
      // 【保留此测试】持续暴露后端缺少循环依赖校验的问题
      await expectBizError(DeptAPI.update(testDeptId, { parentId: childDeptId } as DeptForm), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/dept/{ids} - 删除部门", () => {
    test("正向测试：删除单个部门并验证部门真的被删除", async () => {
      // 创建测试部门
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const deptId = (await DeptAPI.add(form)) as number;

      // 删除部门
      await DeptAPI.deleteByIds(deptId.toString());

      // 删除后查询应返回空（后端返回 data:null，SDK 解析为 undefined）
      const result = await DeptAPI.getFormData(deptId);
      expect(result).toBeUndefined();
    });

    test("正向测试：批量删除多个部门并验证所有部门都被删除", async () => {
      // 创建多个测试部门
      const deptIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
        const deptId = (await DeptAPI.add(form)) as number;
        deptIds.push(deptId);
      }

      // 批量删除
      await DeptAPI.deleteByIds(deptIds.join(","));

      // 删除后查询应返回空
      for (const deptId of deptIds) {
        const result = await DeptAPI.getFormData(deptId);
        expect(result).toBeUndefined();
      }
    });

    test("异常测试：删除不存在的部门", async () => {
      await expectBizError(DeptAPI.deleteByIds("99999999"), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：空的 ID 列表", async () => {
      await expectBizError(DeptAPI.deleteByIds(""), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("业务校验：删除有子部门的部门会级联删除子部门", async () => {
      // 创建父部门
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const parentDeptId = (await DeptAPI.add(parentForm)) as number;

      // 创建子部门
      const childForm = createDeptForm({ parentId: parentDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;

      // 删除父部门
      await DeptAPI.deleteByIds(parentDeptId.toString());

      // 级联删除后查询父子部门均应返回空
      const parentResult = await DeptAPI.getFormData(parentDeptId);
      expect(parentResult).toBeUndefined();

      const childResult = await DeptAPI.getFormData(childDeptId);
      expect(childResult).toBeUndefined();
    });

    test("边界测试：超长部门名称应被拒绝", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: "x".repeat(256) });
      await expectBizError(DeptAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界测试：特殊字符部门名称不应污染存储", async () => {
      const specialName = uniqueName("测试<>&\"'部门");
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: specialName });

      const deptId = (await DeptAPI.add(form)) as number;
      expect(deptId).toBeGreaterThan(0);

      try {
        const formData = await DeptAPI.getFormData(deptId);
        // 特殊字符应被原样保存或转义，不应产生 HTML 标签污染
        expect(typeof formData.name).toBe("string");
        expect(formData.name!.length).toBeGreaterThan(0);
        expect(formData.name).not.toMatch(/<[^>]+>/);
      } finally {
        try {
          await DeptAPI.deleteByIds(deptId.toString());
        } catch {}
      }
    });
  });
});
