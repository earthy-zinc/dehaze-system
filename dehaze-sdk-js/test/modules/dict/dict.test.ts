import { DictAPI, DictForm, DictTypeForm, DictTypePageVO, DictPageVO } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import {
  createDictTypeForm,
  createDictTypeQuery,
  createDictForm,
  createDictQuery,
} from "#/factories/dict";
import { uniqueCode, uniqueName } from "#/factories/common";
import { USERS } from "#/factories/constants";
import { login } from "#/utils/auth";

describe("字典管理接口测试", () => {
  // 统一登记创建的字典类型和字典数据ID，用于 afterAll 清理
  const createdDictTypeIds: number[] = [];
  const createdDictIds: number[] = [];

  async function safeDelete(fn: () => Promise<unknown>): Promise<void> {
    try {
      await fn();
    } catch (e) {
      console.warn(`清理失败:`, e);
    }
  }

  // addDictType 返回 void，无法直接取 ID，需通过列表查询定位新记录
  async function createDictType(form: DictTypeForm): Promise<DictTypePageVO | undefined> {
    await DictAPI.addDictType(form);
    const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ keywords: form.code! }));
    return pageResult.list.find((d) => d.code === form.code);
  }

  // addDict 返回 void，无法直接取 ID，需通过列表查询定位新记录
  async function createDict(form: DictForm): Promise<DictPageVO | undefined> {
    await DictAPI.addDict(form);
    const pageResult = await DictAPI.getDictPage(
      createDictQuery({ typeCode: form.typeCode!, keywords: form.name! })
    );
    return pageResult.list.find((d) => d.name === form.name && d.value === form.value);
  }

  // 断言列表每一项的名称或编码命中搜索关键词
  function expectKeywordMatch(items: DictTypePageVO[], keyword: string): void {
    const lower = keyword.toLowerCase();
    items.forEach((item) => {
      const matchName = item.name!.toLowerCase().includes(lower);
      const matchCode = item.code!.toLowerCase().includes(lower);
      expect(matchName || matchCode).toBe(true);
    });
  }

  afterAll(async () => {
    for (const dictId of createdDictIds) {
      await safeDelete(() => DictAPI.deleteDictByIds(dictId.toString()));
    }
    for (const dictTypeId of createdDictTypeIds) {
      await safeDelete(() => DictAPI.deleteDictTypes(dictTypeId.toString()));
    }
  });

  describe("GET /api/v1/dict/types/page - 字典类型分页列表", () => {
    test("获取字典类型分页列表并验证数据结构", async () => {
      const query = createDictTypeQuery({ pageNum: 1, pageSize: 10 });

      const result = await DictAPI.getDictTypePage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(0);
      expect(result.list.length).toBeLessThanOrEqual(10);

      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(firstItem.id).toBeGreaterThan(0);
        expect(typeof firstItem.code).toBe("string");
        expect(firstItem.code!.length).toBeGreaterThan(0);
        expect(typeof firstItem.name).toBe("string");
        expect(firstItem.name!.length).toBeGreaterThan(0);
      }
    });

    test("按关键词搜索字典类型并验证搜索结果", async () => {
      const allTypes = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 100 }));
      expect(allTypes.list.length).toBeGreaterThan(0);

      const searchKeyword = allTypes.list[0]!.name!.substring(0, 2);

      const result = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: searchKeyword })
      );
      expect(result.list.length).toBeGreaterThan(0);

      expectKeywordMatch(result.list, searchKeyword);
    });

    test("按编码搜索字典类型并验证结果匹配", async () => {
      const allTypes = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 100 }));
      expect(allTypes.list.length).toBeGreaterThan(0);
      const searchCode = allTypes.list[0]!.code!;
      const codePrefix = searchCode.substring(0, Math.min(3, searchCode.length));

      const result = await DictAPI.getDictTypePage(createDictTypeQuery({ keywords: codePrefix }));
      expect(result.list.length).toBeGreaterThan(0);

      expectKeywordMatch(result.list, codePrefix);
    });

    test("边界：特殊字符搜索不引发 XSS 风险", async () => {
      const result = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: "<script>alert(1)</script>" })
      );
      expect(Array.isArray(result.list)).toBe(true);
      expect(JSON.stringify(result)).not.toContain("<script>");
    });

    test("分页逻辑验证 - 不同页码返回不同数据", async () => {
      const pageSize = 5;
      const page1Result = await DictAPI.getDictTypePage(
        createDictTypeQuery({ pageNum: 1, pageSize })
      );
      const page2Result = await DictAPI.getDictTypePage(
        createDictTypeQuery({ pageNum: 2, pageSize })
      );

      expect(page1Result.list.length).toBeLessThanOrEqual(pageSize);
      expect(page2Result.list.length).toBeLessThanOrEqual(pageSize);

      if (page1Result.list.length > 0 && page2Result.list.length > 0) {
        const page1Ids = page1Result.list.map((d) => d.id);
        const page2Ids = page2Result.list.map((d) => d.id);
        const hasIntersection = page1Ids.some((id) => page2Ids.includes(id));
        expect(hasIntersection).toBe(false);
      }
    });
  });

  describe("GET /api/v1/dict/types/{id}/form - 字典类型表单数据", () => {
    test("获取字典类型表单数据并验证数据准确性", async () => {
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const dictType = pageResult.list[0]!;
      const dictTypeId = dictType.id!;

      const result = await DictAPI.getDictTypeForm(dictTypeId);

      expect(result.id).toBe(dictTypeId);
      expect(result.name).toBe(dictType.name);
      expect(result.code).toBe(dictType.code);
      expect(result.status).toBe(dictType.status);
      if (dictType.remark) {
        expect(result.remark).toBe(dictType.remark);
      }
    });

    test("获取不存在字典类型的表单数据应返回业务错误", async () => {
      await expectBizError(DictAPI.getDictTypeForm(99999999), "A0401", "不存在");
    });
  });

  describe("POST /api/v1/dict/types - 新增字典类型", () => {
    test("创建字典类型并验证数据真实持久化", async () => {
      const form = createDictTypeForm({ remark: "这是一个测试字典类型" });

      const createdDictType = await createDictType(form);

      expect(createdDictType).toBeDefined();
      expect(createdDictType?.name).toBe(form.name);
      expect(createdDictType?.status).toBe(form.status);
      expect(createdDictType?.remark).toBe(form.remark);
      if (createdDictType?.id) {
        createdDictTypeIds.push(createdDictType.id);
      }
    });

    test("创建带备注的字典类型并验证备注正确", async () => {
      const testRemark = "这是一个测试字典类型的备注";
      const form = createDictTypeForm({ remark: testRemark });

      const createdDictType = await createDictType(form);

      expect(createdDictType?.remark).toBe(testRemark);
      if (createdDictType?.id) {
        createdDictTypeIds.push(createdDictType.id);
      }
    });

    test("创建禁用状态的字典类型并验证状态值", async () => {
      const form = createDictTypeForm({ status: 0 });

      const createdDictType = await createDictType(form);

      expect(createdDictType?.status).toBe(0);
      if (createdDictType?.id) {
        createdDictTypeIds.push(createdDictType.id);
      }
    });

    test("边界：不传 status 参数时默认为启用(1)", async () => {
      const form: Partial<DictTypeForm> = {
        code: uniqueCode("TEST_TYPE"),
        name: uniqueName("测试默认状态"),
        // 不传 status
      };
      const created = await createDictType(form as DictTypeForm);

      expect(created).toBeDefined();
      expect(created?.status).toBe(1);
      if (created?.id) {
        createdDictTypeIds.push(created.id);
      }
    });

    test("参数校验：缺少必需字段 code", async () => {
      const form: Partial<DictTypeForm> = {
        name: "测试字典类型",
        status: 1,
      };

      await expectBizError(DictAPI.addDictType(form as DictTypeForm), ["B0001", "A0400"]);
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form: Partial<DictTypeForm> = {
        code: "TEST_TYPE",
        status: 1,
      };

      await expectBizError(DictAPI.addDictType(form as DictTypeForm), ["B0001", "A0400"]);
    });

    test("参数校验：字典类型编码已存在", async () => {
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const existingCode = pageResult.list[0]!.code!;
      const form = createDictTypeForm({ code: existingCode });

      await expectBizError(DictAPI.addDictType(form), "A0501", ["编码", "code"]);
    });
  });

  describe("PUT /api/v1/dict/types/{id} - 修改字典类型", () => {
    let testDictTypeId: number;
    let originalForm: DictTypeForm;

    beforeAll(async () => {
      const dictType = await createDictType(createDictTypeForm());
      testDictTypeId = dictType!.id!;
      createdDictTypeIds.push(testDictTypeId);
      originalForm = await DictAPI.getDictTypeForm(testDictTypeId);
    });

    test("更新字典类型名称并验证更新真实生效", async () => {
      const newName = `更新后的字典类型名称_${Date.now()}`;

      // DictTypeForm 的 @NotBlank 要求 name 和 code 必填，必须发送完整表单
      await DictAPI.updateDictType(testDictTypeId, { ...originalForm, name: newName });

      const formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.name).toBe(newName);
    });

    test("更新字典类型状态并验证状态值正确", async () => {
      await DictAPI.updateDictType(testDictTypeId, { ...originalForm, status: 0 });

      let formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.status).toBe(0);

      await DictAPI.updateDictType(testDictTypeId, { ...originalForm, status: 1 });

      formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.status).toBe(1);
    });

    test("更新字典类型备注并验证备注值正确", async () => {
      const newRemark = "更新后的备注信息";

      await DictAPI.updateDictType(testDictTypeId, { ...originalForm, remark: newRemark });

      const formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.remark).toBe(newRemark);
    });

    test("更新不存在的字典类型应返回业务错误", async () => {
      await expectBizError(
        DictAPI.updateDictType(99999999, { ...originalForm, name: "测试" }),
        "A0401",
        "不存在"
      );
    });

    test("参数校验：字典类型编码冲突", async () => {
      // 后端 code 已改为只读字段，任何修改 code 的请求都会被 A0503"字典类型编码不可修改"拒绝
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 100 }));
      const other = pageResult.list.find((d) => d.id !== testDictTypeId);
      expect(other).toBeDefined();
      const existingCode = other!.code!;

      await expectBizError(
        DictAPI.updateDictType(testDictTypeId, { ...originalForm, code: existingCode }),
        "A0503"
      );

      const formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.code).not.toBe(existingCode);
    });

    test("边界：字典类型编码不可修改（只读字段）", async () => {
      const originalCode = originalForm.code;
      const newCode = uniqueCode("ATTEMPT_CHANGE");

      try {
        await DictAPI.updateDictType(testDictTypeId, { ...originalForm, code: newCode });
      } catch {
        // 后端可能拒绝修改 code，成功或失败都不影响后续只读校验
      }

      const formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.code).toBe(originalCode);
    });
  });

  describe("DELETE /api/v1/dict/types/{ids} - 删除字典类型", () => {
    let testDictTypeIds: number[] = [];

    beforeAll(async () => {
      for (let i = 0; i < 3; i++) {
        const dictType = await createDictType(createDictTypeForm());
        if (dictType?.id) {
          testDictTypeIds.push(dictType.id);
          // 不登记到 createdDictTypeIds，因为会在测试中删除
        }
      }
    });

    test("删除单个字典类型并验证真的被删除", async () => {
      const dictTypeId = testDictTypeIds[0];
      await DictAPI.deleteDictTypes(dictTypeId!.toString());
      await expectBizError(DictAPI.getDictTypeForm(dictTypeId!), "A0401", "不存在");
    });

    test("批量删除多个字典类型并验证所有都被删除", async () => {
      const ids = testDictTypeIds.slice(1, 3);

      await DictAPI.deleteDictTypes(ids.join(","));

      for (const dictTypeId of ids) {
        await expectBizError(DictAPI.getDictTypeForm(dictTypeId!), "A0401");
      }
    });

    test("删除不存在的字典类型应返回业务错误", async () => {
      await expectBizError(DictAPI.deleteDictTypes("99999999"), "A0401", "不存在");
    });

    test("边界：删除系统预置字典类型应失败", async () => {
      // 预置类型 create_time 最早，取列表中的最小值作为预置类型
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 100 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const presetType = pageResult.list.reduce((earliest, d) =>
        d.createTime && earliest.createTime && d.createTime < earliest.createTime ? d : earliest
      );
      const presetTypeId = presetType!.id!;

      await expectBizError(DictAPI.deleteDictTypes(presetTypeId.toString()), [
        "A0503",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);

      const formData = await DictAPI.getDictTypeForm(presetTypeId);
      expect(formData.id).toBe(presetTypeId);
    });

    test("参数校验：空的ID列表", async () => {
      // 空字符串在 Gin 路由 :ids 下不会命中（返回 404），invalid 表示非数字 ID
      await expectBizError(DictAPI.deleteDictTypes("invalid"), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("业务校验：不能删除有字典数据的字典类型", async () => {
      const dictTypeForm = createDictTypeForm();
      const dictType = await createDictType(dictTypeForm);
      expect(dictType).toBeDefined();
      const dictTypeId = dictType!.id;
      createdDictTypeIds.push(dictTypeId!);

      const dictForm = createDictForm({ typeCode: dictTypeForm.code || "" });
      const createdDict = await createDict(dictForm);
      if (createdDict?.id) createdDictIds.push(createdDict.id);

      await expectBizError(DictAPI.deleteDictTypes(dictTypeId!.toString()), "A0504", "字典数据");

      const formData = await DictAPI.getDictTypeForm(dictTypeId!);
      expect(formData.id).toBe(dictTypeId);
    });

    test("强制删除：force=true 时级联删除关联的字典数据", async () => {
      const dictTypeForm = createDictTypeForm();
      const dictType = await createDictType(dictTypeForm);
      expect(dictType).toBeDefined();
      const dictTypeId = dictType!.id;

      const dictForm = createDictForm({ typeCode: dictTypeForm.code || "" });
      await DictAPI.addDict(dictForm);

      await DictAPI.deleteDictTypes(dictTypeId!.toString(), true);

      await expectBizError(DictAPI.getDictTypeForm(dictTypeId!), "A0401", "不存在");

      const dictPageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: dictTypeForm.code!, pageSize: 100 })
      );
      expect(dictPageResult.list.length).toBe(0);
    });
  });

  describe("GET /api/v1/dict/{typeCode}/options - 字典下拉列表", () => {
    test("获取字典下拉列表并验证数据格式", async () => {
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const typeCode = pageResult.list[0]!.code!;

      const result = await DictAPI.getDictOptions(typeCode);
      expect(Array.isArray(result)).toBe(true);

      result.forEach((option: any) => {
        expect(option.value).toBeTruthy();
        expect(typeof option.value).toBe("string");
        expect(typeof option.label).toBe("string");
        expect(option.label.length).toBeGreaterThan(0);
      });
    });

    test("获取不存在的字典类型的下拉列表应返回空数组", async () => {
      const nonExistTypeCode = uniqueCode("NON_EXIST");

      const result = await DictAPI.getDictOptions(nonExistTypeCode);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(0);
    });

    test("边界：下拉选项仅返回启用状态的字典数据", async () => {
      const typeForm = createDictTypeForm();
      const dictType = await createDictType(typeForm);
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      const enabledForm = createDictForm({ typeCode: typeForm.code!, status: 1 });
      const disabledForm = createDictForm({ typeCode: typeForm.code!, status: 0 });
      const enabledDict = await createDict(enabledForm);
      const disabledDict = await createDict(disabledForm);
      if (enabledDict?.id) createdDictIds.push(enabledDict.id);
      if (disabledDict?.id) createdDictIds.push(disabledDict.id);

      const options = await DictAPI.getDictOptions(typeForm.code!);
      const optionValues = options.map((o: any) => o.value);
      expect(optionValues).toContain(enabledForm.value);
      expect(optionValues).not.toContain(disabledForm.value);
    });

    test("验证：禁用字典类型后下拉选项不再返回该类型数据", async () => {
      const typeForm = createDictTypeForm({ status: 1 });
      const dictType = await createDictType(typeForm);
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      const dictForm = createDictForm({ typeCode: typeForm.code!, status: 1 });
      const dict = await createDict(dictForm);
      if (dict?.id) createdDictIds.push(dict.id);

      const optionsBefore = await DictAPI.getDictOptions(typeForm.code!);
      expect(optionsBefore.length).toBeGreaterThan(0);

      const typeFormData = await DictAPI.getDictTypeForm(dictType!.id!);
      await DictAPI.updateDictType(dictType!.id!, { ...typeFormData, status: 0 });

      const optionsAfter = await DictAPI.getDictOptions(typeForm.code!);
      expect(optionsAfter.length).toBe(0);
    });
  });

  describe("GET /api/v1/dict/page - 字典分页列表", () => {
    let testTypeCode: string;

    beforeAll(async () => {
      // getDictPage 要求 typeCode 必填，先确保存在一个字典类型
      const typePageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      if (typePageResult.list.length === 0) {
        const dictType = await createDictType(createDictTypeForm());
        testTypeCode = dictType!.code!;
        if (dictType?.id) createdDictTypeIds.push(dictType.id);
      } else {
        testTypeCode = typePageResult.list[0]!.code!;
      }

      // 创建测试字典数据项，确保搜索测试有数据可用
      for (let i = 0; i < 3; i++) {
        const dictForm = createDictForm({ typeCode: testTypeCode });
        const created = await createDict(dictForm);
        if (created?.id) createdDictIds.push(created.id);
      }
    });

    test("获取字典分页列表并验证数据结构", async () => {
      const query = createDictQuery({ typeCode: testTypeCode, pageSize: 10 });

      const result = await DictAPI.getDictPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(0);
      expect(result.list.length).toBeLessThanOrEqual(10);

      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(firstItem.id).toBeGreaterThan(0);
        expect(typeof firstItem.name).toBe("string");
        expect(firstItem.name!.length).toBeGreaterThan(0);
        expect(typeof firstItem.value).toBe("string");
        expect(firstItem.value!.length).toBeGreaterThan(0);
      }
    });

    test("按字典类型编码筛选并验证筛选结果", async () => {
      const result = await DictAPI.getDictPage(createDictQuery({ typeCode: testTypeCode }));

      expect(Array.isArray(result.list)).toBe(true);
    });

    test("按字典名称搜索并验证搜索结果", async () => {
      const allDicts = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, pageSize: 100 })
      );
      expect(allDicts.list.length).toBeGreaterThan(0);
      const firstDict = allDicts.list[0]!;
      const searchKeyword = firstDict.name!.substring(0, 1);

      const result = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, keywords: searchKeyword })
      );
      expect(result.list.length).toBeGreaterThan(0);

      result.list.forEach((item) => {
        expect(item.name!.toLowerCase()).toContain(searchKeyword.toLowerCase());
      });
    });

    test("验证：字典数据按 sort 字段升序返回", async () => {
      const result = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, pageSize: 100 })
      );
      if (result.list.length > 1) {
        for (let i = 1; i < result.list.length; i++) {
          expect(result.list[i]!.sort!).toBeGreaterThanOrEqual(result.list[i - 1]!.sort!);
        }
      }
    });
  });

  describe("GET /api/v1/dict/{id}/form - 字典数据表单数据", () => {
    test("获取字典数据表单数据并验证数据准确性", async () => {
      // 找一个存在字典数据的类型，用于获取其下的字典表单
      const typePageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 100 }));
      expect(typePageResult.list.length).toBeGreaterThan(0);

      let dictPage: { list: any[]; total: number } | null = null;
      for (const dictType of typePageResult.list) {
        const page = await DictAPI.getDictPage(
          createDictQuery({ typeCode: dictType.code!, pageSize: 1 })
        );
        if (page.list.length > 0) {
          dictPage = page;
          break;
        }
      }
      expect(dictPage).not.toBeNull();
      const dict = dictPage!.list[0]!;
      const dictId = dict.id!;

      const result = await DictAPI.getDictFormData(dictId);

      expect(result.id).toBe(dictId);
      expect(result.name).toBe(dict.name);
      expect(result.value).toBe(dict.value);
      expect(result.status).toBe(dict.status);
      expect(typeof result.sort).toBe("number");
    });

    test("获取不存在字典数据的表单数据应返回业务错误", async () => {
      await expectBizError(DictAPI.getDictFormData(99999999), "A0401", "不存在");
    });
  });

  describe("POST /api/v1/dict - 新增字典", () => {
    let testTypeCode: string;

    beforeAll(async () => {
      const dictType = await createDictType(createDictTypeForm());
      testTypeCode = dictType!.code!;
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }
    });

    test("创建字典并验证数据真实持久化", async () => {
      const testRemark = "这是一个测试字典";
      const form = createDictForm({ typeCode: testTypeCode, remark: testRemark });

      const createdDict = await createDict(form);

      expect(createdDict).toBeDefined();
      expect(createdDict?.name).toBe(form.name);
      expect(createdDict?.value).toBe(form.value);
      expect(createdDict?.status).toBe(form.status);

      if (createdDict?.id) {
        createdDictIds.push(createdDict.id);
        const formData = await DictAPI.getDictFormData(createdDict.id);
        expect(formData.name).toBe(form.name);
        expect(formData.value).toBe(form.value);
        expect(formData.typeCode).toBe(testTypeCode);
        expect(formData.sort).toBe(form.sort);
        expect(formData.status).toBe(form.status);
        expect(formData.remark).toBe(testRemark);
      }
    });

    test("创建带备注的字典并验证备注正确", async () => {
      const testRemark = "这是一个测试字典的备注";
      const form = createDictForm({ typeCode: testTypeCode, sort: 2, remark: testRemark });

      const createdDict = await createDict(form);

      expect(createdDict).toBeDefined();

      if (createdDict?.id) {
        createdDictIds.push(createdDict.id);
        const formData = await DictAPI.getDictFormData(createdDict.id);
        expect(formData.remark).toBe(testRemark);
      }
    });

    test("创建禁用状态的字典并验证状态值", async () => {
      const form = createDictForm({ typeCode: testTypeCode, sort: 3, status: 0 });

      const createdDict = await createDict(form);

      expect(createdDict).toBeDefined();
      expect(createdDict?.status).toBe(0);

      if (createdDict?.id) {
        createdDictIds.push(createdDict.id);
        const formData = await DictAPI.getDictFormData(createdDict.id);
        expect(formData.status).toBe(0);
      }
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form: Partial<DictForm> = {
        value: "1",
        typeCode: testTypeCode,
        status: 1,
      };

      await expectBizError(DictAPI.addDict(form as DictForm), "A0400");
    });

    test("参数校验：缺少必需字段 value", async () => {
      const form: Partial<DictForm> = {
        name: "测试字典",
        typeCode: testTypeCode,
        status: 1,
      };

      await expectBizError(DictAPI.addDict(form as DictForm), "A0400");
    });

    test("参数校验：缺少必需字段 typeCode", async () => {
      const form: Partial<DictForm> = {
        name: "测试字典",
        value: "1",
        status: 1,
      };
      await expectBizError(DictAPI.addDict(form as DictForm), "A0400");
    });

    test("参数校验：字典类型编码不存在应抛出业务错误", async () => {
      const form = createDictForm({ typeCode: uniqueCode("NON_EXIST") });

      await expectBizError(DictAPI.addDict(form), "A0401", ["类型", "typeCode"]);
    });

    test("参数校验：同一类型下字典值已存在应失败", async () => {
      const form = createDictForm({ typeCode: testTypeCode });
      const created = await createDict(form);
      if (created?.id) {
        createdDictIds.push(created.id);
      }

      const dupForm = createDictForm({ typeCode: testTypeCode, value: form.value! });
      await expectBizError(DictAPI.addDict(dupForm), [
        "A0501",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：字典标签为空应失败", async () => {
      const form: Partial<DictForm> = {
        name: "",
        value: "test_value",
        typeCode: testTypeCode,
        status: 1,
      };
      await expectBizError(DictAPI.addDict(form as DictForm), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：字典值为空应失败", async () => {
      const form: Partial<DictForm> = {
        name: "测试字典",
        value: "",
        typeCode: testTypeCode,
        status: 1,
      };
      await expectBizError(DictAPI.addDict(form as DictForm), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：不传 sort 参数时默认值为 1", async () => {
      const form: Partial<DictForm> = {
        name: uniqueName("测试默认排序"),
        value: Date.now().toString().slice(-6),
        typeCode: testTypeCode,
        status: 1,
        // 不传 sort
      };
      const created = await createDict(form as DictForm);

      expect(created).toBeDefined();
      if (created?.id) {
        createdDictIds.push(created.id);
        const formData = await DictAPI.getDictFormData(created.id);
        expect(formData.sort).toBe(1);
      }
    });
  });

  describe("PUT /api/v1/dict/{id} - 修改字典", () => {
    let testDictId: number;
    let testTypeCode: string;
    let originalDictForm: DictForm;

    beforeAll(async () => {
      const dictType = await createDictType(createDictTypeForm());
      testTypeCode = dictType!.code;
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      const dictForm = createDictForm({ typeCode: testTypeCode });
      const createdDict = await createDict(dictForm);
      testDictId = createdDict!.id!;
      createdDictIds.push(testDictId);
      originalDictForm = await DictAPI.getDictFormData(testDictId);
    });

    test("更新字典名称并验证更新真实生效", async () => {
      const newName = `更新后的字典名称_${Date.now()}`;

      // DictForm 的 @NotBlank 要求 typeCode/name/value 必填，必须发送完整表单
      await DictAPI.updateDict(testDictId, { ...originalDictForm, name: newName });

      const formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.name).toBe(newName);
      expect(formData.typeCode).toBe(testTypeCode);
    });

    test("更新字典值并验证值正确", async () => {
      const newValue = Date.now().toString().slice(-6);

      await DictAPI.updateDict(testDictId, { ...originalDictForm, value: newValue });

      const formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.value).toBe(newValue);
    });

    test("更新字典状态并验证状态值正确", async () => {
      await DictAPI.updateDict(testDictId, { ...originalDictForm, status: 0 });

      let formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.status).toBe(0);

      await DictAPI.updateDict(testDictId, { ...originalDictForm, status: 1 });

      formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.status).toBe(1);
    });

    test("更新字典排序并验证排序值正确", async () => {
      const newSort = 999;

      await DictAPI.updateDict(testDictId, { ...originalDictForm, sort: newSort });

      const formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.sort).toBe(newSort);
    });

    test("更新字典备注并验证备注值正确", async () => {
      const newRemark = "更新后的备注信息";

      await DictAPI.updateDict(testDictId, { ...originalDictForm, remark: newRemark });

      const formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.remark).toBe(newRemark);
    });

    test("更新不存在的字典应返回业务错误", async () => {
      await expectBizError(
        DictAPI.updateDict(99999999, { ...originalDictForm, name: "测试" }),
        "A0401",
        "不存在"
      );
    });

    test("边界：typeCode 只读，修改 typeCode 应失败或保持不变", async () => {
      const originalTypeCode = originalDictForm.typeCode;
      const newTypeCode = uniqueCode("ATTEMPT_CHANGE");

      try {
        await DictAPI.updateDict(testDictId, { ...originalDictForm, typeCode: newTypeCode });
      } catch {
        // 后端可能拒绝修改 typeCode，成功或失败都不影响后续只读校验
      }

      const formData = await DictAPI.getDictFormData(testDictId);
      expect(formData.typeCode).toBe(originalTypeCode);
    });

    test("参数校验：修改后字典值与同类型下其他字典重复应失败", async () => {
      const otherForm = createDictForm({ typeCode: testTypeCode });
      const otherDict = await createDict(otherForm);
      if (otherDict?.id) {
        createdDictIds.push(otherDict.id);
      }

      await expectBizError(
        DictAPI.updateDict(testDictId, { ...originalDictForm, value: otherForm.value! }),
        ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });
  });

  describe("DELETE /api/v1/dict/{ids} - 删除字典", () => {
    let testDictIds: number[] = [];
    let testTypeCode: string;

    beforeAll(async () => {
      const dictType = await createDictType(createDictTypeForm());
      testTypeCode = dictType!.code;
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      for (let i = 0; i < 3; i++) {
        const form = createDictForm({ typeCode: testTypeCode, sort: i + 1 });
        await DictAPI.addDict(form);
      }

      const pageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, pageSize: 100 })
      );
      testDictIds = pageResult.list.map((dict) => dict.id!);
    });

    test("删除单个字典并验证字典真的被删除", async () => {
      const dictId = testDictIds[0];
      await DictAPI.deleteDictByIds(dictId!.toString());
      await expectBizError(DictAPI.getDictFormData(dictId!), "A0401", "不存在");
    });

    test("批量删除多个字典并验证所有字典都被删除", async () => {
      const ids = testDictIds.slice(1);

      await DictAPI.deleteDictByIds(ids.join(","));

      for (const dictId of ids) {
        await expectBizError(DictAPI.getDictFormData(dictId!), "A0401");
      }
    });

    test("删除不存在的字典应返回业务错误", async () => {
      await expectBizError(DictAPI.deleteDictByIds("99999999"), "A0401", "不存在");
    });

    test("参数校验：空的ID列表", async () => {
      // 空字符串在 Gin 路由 :ids 下不会命中（返回 404），invalid 表示非数字 ID
      await expectBizError(DictAPI.deleteDictByIds("invalid"), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户新增字典类型应失败", async () => {
      const form = createDictTypeForm();
      await expectBizError(DictAPI.addDictType(form), [
        "A0403",
        "A0301",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户修改字典类型应失败", async () => {
      const form = createDictTypeForm();
      await expectBizError(DictAPI.updateDictType(1, form), [
        "A0403",
        "A0301",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户删除字典类型应失败", async () => {
      await expectBizError(DictAPI.deleteDictTypes("1"), [
        "A0403",
        "A0301",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
