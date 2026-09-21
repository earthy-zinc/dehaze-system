import { DeptAPI, DeptForm } from "../../../index";
import { DeptVO } from "@/api/dept/model";
import { expectBizError } from "#/utils/assertion";
import { createDeptForm, createDeptQuery, createDeptChain } from "#/factories/dept";
import { uniqueName } from "#/factories/common";
import { DEPTS, USERS } from "#/factories/constants";
import { login } from "#/utils/auth";

/** 在树形结构中递归查找部门节点，未找到返回 undefined */
function findDeptInTree(tree: DeptVO[], id: number): DeptVO | undefined {
  for (const dept of tree) {
    if (dept.id === id) return dept;
    if (dept.children) {
      const found = findDeptInTree(dept.children, id);
      if (found) return found;
    }
  }
  return undefined;
}

/** 收集树中所有启用部门的 ID */
function getEnabledDeptIds(tree: DeptVO[]): number[] {
  const ids: number[] = [];
  const walk = (depts: DeptVO[]) => {
    depts.forEach((dept) => {
      if (dept.status === 1) ids.push(dept.id!);
      if (dept.children?.length) walk(dept.children);
    });
  };
  walk(tree);
  return ids;
}

/** 清理部门：从后往前删除（先删子部门再删父部门），失败静默忽略 */
async function deleteDeptsSafe(ids: number[]): Promise<void> {
  for (const id of ids.reverse()) {
    try {
      await DeptAPI.deleteByIds(id.toString());
    } catch {
      /* 清理失败静默忽略（资源可能已被测试本身删除） */
    }
  }
}

describe("部门管理接口测试", () => {
  describe("GET /api/v1/dept - 部门树形表格", () => {
    test("正向测试：获取部门树形列表并验证树形结构", async () => {
      const result = await DeptAPI.getList();

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const verifyDept = (depts: typeof result) => {
        depts.forEach((dept) => {
          expect(dept.id).toBeGreaterThan(0);
          expect(dept.name).toBeTruthy();
          expect(dept.parentId).toBeGreaterThanOrEqual(0);

          if (dept.children && dept.children.length > 0) {
            dept.children.forEach((child) => {
              expect(child.parentId).toBe(dept.id);
            });
            verifyDept(dept.children);
          }
        });
      };
      verifyDept(result);

      expect(findDeptInTree(result, DEPTS.CQUPT.id)).toBeDefined();
    });

    test("正向测试：按关键词搜索部门并验证搜索结果", async () => {
      const searchKeyword = DEPTS.CQUPT.name.substring(0, 2);
      const query = createDeptQuery({ keywords: searchKeyword });
      const result = await DeptAPI.getList(query);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

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

      expect(Array.isArray(result)).toBe(true);

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

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const rootDepts = result.filter((dept) => dept.parentId === 0);
      expect(rootDepts.length).toBeGreaterThan(0);

      rootDepts.forEach((dept) => {
        expect(dept.parentId).toBe(0);
        expect(dept.id).toBeGreaterThan(0);
        expect(dept.name).toBeTruthy();
      });

      const cquptDept = rootDepts.find((d) => d.id === DEPTS.CQUPT.id);
      expect(cquptDept).toBeDefined();
      expect(cquptDept!.name).toBe(DEPTS.CQUPT.name);
    });
  });

  describe("GET /api/v1/dept/options - 部门下拉列表", () => {
    test("正向测试：获取部门下拉列表并验证数据准确性", async () => {
      const result = await DeptAPI.getOptions();

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      result.forEach((option) => {
        expect(option.value).toBeGreaterThan(0);
        expect(option.label).toBeTruthy();
      });

      const optionIds = result.map((opt) => opt.value);
      expect(optionIds).toContain(DEPTS.CQUPT.id);
    });

    test("正向测试：验证下拉列表只包含启用部门", async () => {
      const options = await DeptAPI.getOptions();
      const treeResult = await DeptAPI.getList();

      expect(options.length).toBeGreaterThan(0);
      expect(treeResult.length).toBeGreaterThan(0);

      const enabledDeptIds = getEnabledDeptIds(treeResult);
      const optionIds = options.map((opt) => opt.value);

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
      await deleteDeptsSafe(createdDeptIds);
    });

    test("正向测试：创建部门并验证数据真实持久化", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });

      const deptId = await DeptAPI.add(form);

      expect(deptId).toBeGreaterThan(0);
      createdDeptIds.push(deptId as number);

      const formData = await DeptAPI.getFormData(deptId as number);
      expect(formData.id).toBe(deptId);
      expect(formData.name).toBe(form.name);
      expect(formData.parentId).toBe(form.parentId);
      expect(formData.sort).toBe(form.sort);
      expect(formData.status).toBe(form.status);
    });

    test("正向测试：创建子部门并验证父子关系", async () => {
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const parentDeptId = (await DeptAPI.add(parentForm)) as number;
      createdDeptIds.push(parentDeptId);

      const childForm = createDeptForm({ parentId: parentDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;
      createdDeptIds.push(childDeptId);

      const childFormData = await DeptAPI.getFormData(childDeptId);
      expect(childFormData.parentId).toBe(parentDeptId);

      const deptList = await DeptAPI.getList();
      const createdChild = findDeptInTree(deptList, childDeptId);
      expect(createdChild).toBeDefined();
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
      // 【保留此测试】持续暴露后端缺少必填字段 name 校验的问题（后端 bug）
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

      const deptId = (await DeptAPI.add(form)) as number;
      createdDeptIds.push(deptId);

      await expectBizError(DeptAPI.add({ ...form, sort: form.sort + 1 }), [
        "A0501",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：父部门不存在应抛出业务错误", async () => {
      // 【保留此测试】持续暴露后端缺少父部门存在性校验的问题（后端 bug）
      const form = createDeptForm({ parentId: 99999999 });

      await expectBizError(DeptAPI.add(form), ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界：在4级部门下新增第5级部门应成功（达到最大层级）", async () => {
      // CQUPT (id=1, 第1级) → 新建3级 → chainIds[2] 是第4级部门
      const chainIds = await createDeptChain(DEPTS.CQUPT.id, 3);
      createdDeptIds.push(...chainIds);

      const form = createDeptForm({ parentId: chainIds[2]! });
      const deptId = await DeptAPI.add(form);
      expect(deptId).toBeGreaterThan(0);
      createdDeptIds.push(deptId as number);
    });

    test("边界：在5级部门下新增第6级部门应失败（超出层级限制）", async () => {
      // CQUPT (id=1, 第1级) → 新建4级 → chainIds[3] 是第5级部门
      const chainIds = await createDeptChain(DEPTS.CQUPT.id, 4);
      createdDeptIds.push(...chainIds);

      const form = createDeptForm({ parentId: chainIds[3]! });
      await expectBizError(DeptAPI.add(form), ["A0504", "A0400", "B0001", "ERR_BAD_REQUEST"]);
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
      createdDeptIds.push(deptId);

      // 特殊字符应被原样保存或转义，不应产生 HTML 标签污染
      const formData = await DeptAPI.getFormData(deptId);
      expect(typeof formData.name).toBe("string");
      expect(formData.name!.length).toBeGreaterThan(0);
      expect(formData.name).not.toMatch(/<[^>]+>/);
    });
  });

  describe("PUT /api/v1/dept/{id} - 修改部门", () => {
    let testDeptId: number;
    let originalDept: DeptForm;
    const additionalDeptIds: number[] = [];

    beforeAll(async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      testDeptId = (await DeptAPI.add(form)) as number;
      originalDept = await DeptAPI.getFormData(testDeptId);
    });

    afterAll(async () => {
      await deleteDeptsSafe([testDeptId, ...additionalDeptIds]);
    });

    test("正向测试：更新部门名称并验证更新真实生效", async () => {
      const newForm = createDeptForm({ parentId: originalDept.parentId });

      await DeptAPI.update(testDeptId, newForm);

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
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const newParentId = (await DeptAPI.add(parentForm)) as number;
      additionalDeptIds.push(newParentId);

      const form: Partial<DeptForm> = {
        name: originalDept.name ?? "",
        parentId: newParentId,
      };
      await DeptAPI.update(testDeptId, form as DeptForm);

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

    test("参数校验：不能将部门的上级设置为其子部门（循环引用检测）", async () => {
      const childForm = createDeptForm({ parentId: testDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;
      additionalDeptIds.push(childDeptId);

      // 循环引用检测（A0503 操作不允许）
      await expectBizError(DeptAPI.update(testDeptId, { parentId: childDeptId } as DeptForm), [
        "A0503",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：不能选择自身作为上级部门", async () => {
      await expectBizError(
        DeptAPI.update(testDeptId, {
          name: originalDept.name ?? "",
          parentId: testDeptId,
        } as DeptForm),
        ["A0503", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("边界：移动部门至超深层级应失败（超出5级限制）", async () => {
      // CQUPT(第1级) + 4级 = chainIds[3] 是第5级
      const chainIds = await createDeptChain(DEPTS.CQUPT.id, 4);
      additionalDeptIds.push(...chainIds);

      // 创建独立的第2级部门，移动到第5级下会变成第6级，应失败
      const targetForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const targetId = (await DeptAPI.add(targetForm)) as number;
      additionalDeptIds.push(targetId);

      await expectBizError(
        DeptAPI.update(targetId, { name: targetForm.name, parentId: chainIds[3]! } as DeptForm),
        ["A0504", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("边界：修改根部门上级应失败（根部门保护）", async () => {
      // 根部门 id=1 不可修改 parentId
      await expectBizError(
        DeptAPI.update(DEPTS.CQUPT.id, {
          name: DEPTS.CQUPT.name,
          parentId: DEPTS.SOFTWARE.id,
        } as DeptForm),
        ["A0503", "A0234", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });
  });

  describe("DELETE /api/v1/dept/{ids} - 删除部门", () => {
    test("正向测试：删除单个部门并验证部门真的被删除", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const deptId = (await DeptAPI.add(form)) as number;

      await DeptAPI.deleteByIds(deptId.toString());

      // 删除后返回 data:null，SDK 解析为 undefined
      const result = await DeptAPI.getFormData(deptId);
      expect(result).toBeUndefined();
    });

    test("正向测试：批量删除多个部门并验证所有部门都被删除", async () => {
      const deptIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
        const deptId = (await DeptAPI.add(form)) as number;
        deptIds.push(deptId);
      }

      await DeptAPI.deleteByIds(deptIds.join(","));

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

    // 空 ID 列表由 SDK 前置校验拦截（DELETE /{ids} 路由无法接收空路径参数，后端 405 是 HTTP 语义正确行为）
    test("参数校验：空的 ID 列表由 SDK 前置校验拦截", async () => {
      await expect(DeptAPI.deleteByIds("")).rejects.toThrow("不能为空");
    });

    test("业务校验：删除有子部门的部门应被拒绝（不级联删除）", async () => {
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const parentDeptId = (await DeptAPI.add(parentForm)) as number;

      const childForm = createDeptForm({ parentId: parentDeptId });
      const childDeptId = (await DeptAPI.add(childForm)) as number;

      // 存在子部门时禁止删除（A0502）
      await expectBizError(DeptAPI.deleteByIds(parentDeptId.toString()), [
        "A0502",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);

      await deleteDeptsSafe([childDeptId, parentDeptId]);
    });

    test("业务校验：删除有关联用户的部门应失败", async () => {
      // DEPTS.SOFTWARE (id=2) 下有预置关联用户（admin、dept_admin、user 等）
      await expectBizError(DeptAPI.deleteByIds(DEPTS.SOFTWARE.id.toString()), [
        "A0502",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：删除根部门应失败（根部门保护）", async () => {
      await expectBizError(DeptAPI.deleteByIds(DEPTS.CQUPT.id.toString()), [
        "A0503",
        "A0234",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("验证：逻辑删除后部门不在列表中展示", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      const deptId = (await DeptAPI.add(form)) as number;

      const listBefore = await DeptAPI.getList();
      expect(findDeptInTree(listBefore, deptId)).toBeDefined();

      await DeptAPI.deleteByIds(deptId.toString());

      const listAfter = await DeptAPI.getList();
      expect(findDeptInTree(listAfter, deptId)).toBeUndefined();
    });

    test("验证：禁用父部门不影响子部门状态（不级联）", async () => {
      const parentForm = createDeptForm({ parentId: DEPTS.CQUPT.id, status: 1 });
      const parentDeptId = (await DeptAPI.add(parentForm)) as number;
      const childForm = createDeptForm({ parentId: parentDeptId, status: 1 });
      const childDeptId = (await DeptAPI.add(childForm)) as number;

      try {
        await DeptAPI.update(parentDeptId, { ...parentForm, status: 0 } as DeptForm);

        const childFormData = await DeptAPI.getFormData(childDeptId);
        expect(childFormData.status).toBe(1);
      } finally {
        await deleteDeptsSafe([childDeptId, parentDeptId]);
      }
    });
  });

  describe("对抗性语料与边界强化", () => {
    const createdDeptIds: number[] = [];

    afterAll(async () => {
      await deleteDeptsSafe(createdDeptIds);
    });

    // 脏语料特征保留 + unique 后缀：名称同级唯一含软删行（T-DPT-035b），固定名称跨运行必撞 A0501
    const dirtyNames: Array<[string, string]> = [
      ["全半角混合", uniqueName("测试Ｔｅｓｔ１２３部⻔")],
      ["CRLF 换行", uniqueName("部\r\n门")],
      ["BOM 前缀", `\uFEFF${uniqueName("部门BOM")}`],
      ["零宽字符", uniqueName("部\u200B门\u200BZWS")],
      ["emoji", uniqueName("部😀门🚀emoji")],
      ["SQL 注入单引号", uniqueName("admin'--部")],
      ["SQL 注入语句", uniqueName("x'; DROP TABLE sys_dept;--")],
    ];

    test.each(dirtyNames)("脏语料往返保真：%s", async (_label, name) => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name });
      const deptId = (await DeptAPI.add(form)) as number;
      expect(deptId).toBeGreaterThan(0);
      createdDeptIds.push(deptId);

      const formData = await DeptAPI.getFormData(deptId);
      // 原样存储往返一致，不允许被静默截断、转义或清洗
      expect(formData.name).toBe(name);
    });

    test("边界：64 字符名称（上限值）应成功", async () => {
      // 唯一化尾缀：名称同级含软删唯一，固定 64 字符跨运行必撞 A0501（角色模块同教训）
      const stamp = Date.now().toString().slice(-8);
      const name = "边".repeat(64 - stamp.length) + stamp;
      const deptId = (await DeptAPI.add(
        createDeptForm({ parentId: DEPTS.CQUPT.id, name })
      )) as number;
      expect(deptId).toBeGreaterThan(0);
      createdDeptIds.push(deptId);

      const formData = await DeptAPI.getFormData(deptId);
      expect(formData.name).toBe(name);
    });

    test("边界：65 字符名称应被拒绝（A0400）", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, name: "边".repeat(65) });
      await expectBizError(DeptAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界：sort=0 应被拒绝（实现口径 sort>=1，T-DPT-046 排序须为正整数）", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id, sort: 0 });
      await expectBizError(DeptAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("唯一性口径：不同父部门下同名允许（同级唯一，对齐文档 T-DPT-009 与 Go 实现）", async () => {
      const sharedName = uniqueName("跨级同名部门");
      const deptId1 = (await DeptAPI.add(
        createDeptForm({ parentId: DEPTS.CQUPT.id, name: sharedName })
      )) as number;
      createdDeptIds.push(deptId1);

      const deptId2 = (await DeptAPI.add(
        createDeptForm({ parentId: DEPTS.SOFTWARE.id, name: sharedName })
      )) as number;
      expect(deptId2).toBeGreaterThan(0);
      createdDeptIds.push(deptId2);
    });

    test("软删语义：同级软删后同名重建被拒（唯一性含已删除记录，T-DPT-035b）", async () => {
      const name = uniqueName("软删复用部门");
      const deptId = (await DeptAPI.add(
        createDeptForm({ parentId: DEPTS.CQUPT.id, name })
      )) as number;
      await DeptAPI.deleteByIds(deptId.toString());

      // 换 sort 改变 body：同 body 快速重建会被 A0002 防重复提交拦截，绕不到唯一性校验
      await expectBizError(
        DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id, name, sort: 101 })),
        ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("【暴露缺陷】批量删除混合存在/不存在 ID 应整体拒绝而非部分删除", async () => {
      // Java 契约：任一 ID 不存在 → A0401 整体失败不误删；
      // Python 现行为：静默忽略不存在 ID 并删除其余 → 误删风险。
      // 期望 A0401（对齐 Java），在 Python 修复前此用例失败即为缺陷暴露信号。
      const deptId = (await DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id }))) as number;

      await expectBizError(DeptAPI.deleteByIds(`${deptId},99999999`), ["A0401"]);

      // 若后端修复为整体失败，部门仍在；清理容错
      createdDeptIds.push(deptId);
    });

    test("缓存一致性：新增部门后下拉选项立即包含新部门", async () => {
      const deptId = (await DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id }))) as number;
      createdDeptIds.push(deptId);

      const options = await DeptAPI.getOptions();
      const optionIds: number[] = [];
      const walk = (nodes: typeof options) => {
        nodes.forEach((n) => {
          optionIds.push(n.value);
          if (n.children?.length) walk(n.children);
        });
      };
      walk(options);
      expect(optionIds).toContain(deptId);
    });

    test("树不变量：移动非叶子部门后父子关系保持完整", async () => {
      const aId = (await DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id }))) as number;
      const bId = (await DeptAPI.add(createDeptForm({ parentId: aId }))) as number;
      const pId = (await DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id }))) as number;
      createdDeptIds.push(pId, bId, aId);

      await DeptAPI.update(aId, {
        name: (await DeptAPI.getFormData(aId)).name!,
        parentId: pId,
      } as DeptForm);

      const tree = await DeptAPI.getList();
      const movedA = findDeptInTree(tree, aId);
      const movedB = findDeptInTree(tree, bId);
      expect(movedA).toBeDefined();
      expect(movedA!.parentId).toBe(pId);
      // 子部门 parent_id 链未断裂，仍可从根遍历到
      expect(movedB).toBeDefined();
      expect(movedB!.parentId).toBe(aId);
    });

    test("性能烟测：20 部门同层下列表查询 < 500ms", async () => {
      const ids: number[] = [];
      for (let i = 0; i < 20; i++) {
        ids.push((await DeptAPI.add(createDeptForm({ parentId: DEPTS.CQUPT.id }))) as number);
      }
      createdDeptIds.push(...ids);

      const start = Date.now();
      const tree = await DeptAPI.getList();
      const elapsed = Date.now() - start;
      expect(Array.isArray(tree)).toBe(true);
      expect(elapsed).toBeLessThan(500);
    });
  });

  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户新增部门应失败", async () => {
      const form = createDeptForm({ parentId: DEPTS.CQUPT.id });
      // 后端 require_permission 对无权限返回 403 + A0301（访问未授权）
      await expectBizError(DeptAPI.add(form), ["A0301"]);
    });

    test("边界：普通用户修改部门应失败", async () => {
      await expectBizError(DeptAPI.update(DEPTS.SOFTWARE.id, { name: "hacked" } as any), [
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户删除部门应失败", async () => {
      // 后端 require_permission 对无权限返回 403 + A0301（访问未授权）
      await expectBizError(DeptAPI.deleteByIds(DEPTS.SOFTWARE.id.toString()), ["A0301"]);
    });
  });
});
