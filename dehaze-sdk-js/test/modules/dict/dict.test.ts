import { DictAPI, DictForm, DictQuery, DictTypeForm, DictTypeQuery } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import {
  createDictTypeForm,
  createDictTypeQuery,
  createDictForm,
  createDictQuery,
} from "#/factories/dict";
import { uniqueCode } from "#/factories/common";

describe("字典管理接口测试", () => {
  // 统一管理创建的字典类型和字典数据ID，用于清理
  const createdDictTypeIds: number[] = [];
  const createdDictIds: number[] = [];
  afterAll(async () => {
    for (const dictId of createdDictIds) {
      try {
        await DictAPI.deleteDictByIds(dictId.toString());
      } catch (e) {}
    }
    for (const dictTypeId of createdDictTypeIds) {
      try {
        await DictAPI.deleteDictTypes(dictTypeId.toString());
      } catch (e) {}
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

      result.list.forEach((item) => {
        const matchName = item.name!.toLowerCase().includes(searchKeyword.toLowerCase());
        const matchCode = item.code!.toLowerCase().includes(searchKeyword.toLowerCase());
        expect(matchName || matchCode).toBe(true);
      });
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

    test("超大页码应返回空数组", async () => {
      const result = await DictAPI.getDictTypePage(
        createDictTypeQuery({ pageNum: 99999, pageSize: 10 })
      );

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBe(0);
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

      // addDictType 返回 undefined（后端 Result<Void>，data:null 被 Jackson 省略）
      await DictAPI.addDictType(form);

      const pageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: form.code! })
      );
      const createdDictType = pageResult.list.find((dictType) => dictType.code === form.code);
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

      await DictAPI.addDictType(form);

      const pageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: form.code! })
      );
      const createdDictType = pageResult.list.find((dictType) => dictType.code === form.code);
      expect(createdDictType?.remark).toBe(testRemark);
      if (createdDictType?.id) {
        createdDictTypeIds.push(createdDictType.id);
      }
    });

    test("创建禁用状态的字典类型并验证状态值", async () => {
      const form = createDictTypeForm({ status: 0 });

      await DictAPI.addDictType(form);

      const pageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: form.code! })
      );
      const createdDictType = pageResult.list.find((dictType) => dictType.code === form.code);
      expect(createdDictType?.status).toBe(0);
      if (createdDictType?.id) {
        createdDictTypeIds.push(createdDictType.id);
      }
    });

    test("参数校验：缺少必需字段 code", async () => {
      const form: Partial<DictTypeForm> = {
        name: "测试字典类型",
        status: 1,
      };

      await expectBizError(
        DictAPI.addDictType(form as DictTypeForm),
        ["B0001", "A0400"],
        undefined,
        true
      );
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form: Partial<DictTypeForm> = {
        code: "TEST_TYPE",
        status: 1,
      };

      await expectBizError(
        DictAPI.addDictType(form as DictTypeForm),
        ["B0001", "A0400"],
        undefined,
        true
      );
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
      const form = createDictTypeForm();
      await DictAPI.addDictType(form);

      const pageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: form.code! })
      );
      const createdDictType = pageResult.list.find((dictType) => dictType.code === form.code);
      testDictTypeId = createdDictType!.id!;
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
      const pageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      expect(pageResult.list.length).toBeGreaterThan(0);
      const existingCode = pageResult.list[0]!.code!;

      await expectBizError(
        DictAPI.updateDictType(testDictTypeId, { ...originalForm, code: existingCode }),
        "A0501"
      );

      const formData = await DictAPI.getDictTypeForm(testDictTypeId);
      expect(formData.code).not.toBe(existingCode);
    });
  });

  describe("DELETE /api/v1/dict/types/{ids} - 删除字典类型", () => {
    let testDictTypeIds: number[] = [];

    beforeAll(async () => {
      for (let i = 0; i < 3; i++) {
        const form = createDictTypeForm();
        await DictAPI.addDictType(form);

        const pageResult = await DictAPI.getDictTypePage(
          createDictTypeQuery({ keywords: form.code! })
        );
        const createdDictType = pageResult.list.find((dictType) => dictType.code === form.code);
        if (createdDictType?.id) {
          testDictTypeIds.push(createdDictType.id);
          // 注意：这里不加入 createdDictTypeIds，因为会在测试中删除
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

    test("参数校验：空的ID列表", async () => {
      // 空字符串在 Gin 路由 :ids 下不会命中（返回 404），invalid 表示非数字 ID
      await expectBizError(
        DictAPI.deleteDictTypes("invalid"),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("业务校验：不能删除有字典数据的字典类型", async () => {
      const dictTypeForm = createDictTypeForm();
      await DictAPI.addDictType(dictTypeForm);
      const dictTypePageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: dictTypeForm.code! })
      );
      const dictType = dictTypePageResult.list.find((d) => d.code === dictTypeForm.code);
      expect(dictType).toBeDefined();
      const dictTypeId = dictType!.id;
      createdDictTypeIds.push(dictTypeId!);

      const dictForm = createDictForm({
        typeCode: dictTypeForm.code || "",
      });
      await DictAPI.addDict(dictForm);

      await expectBizError(DictAPI.deleteDictTypes(dictTypeId!.toString()), "A0504", "字典数据");

      // 验证字典类型仍存在
      const formData = await DictAPI.getDictTypeForm(dictTypeId!);
      expect(formData.id).toBe(dictTypeId);

      // 清理测试数据
      try {
        const dictPageResult = await DictAPI.getDictPage(
          createDictQuery({ typeCode: dictTypeForm.code!, pageSize: 1000 })
        );
        if (dictPageResult.list.length > 0) {
          const dictIds = dictPageResult.list.map((d) => d.id).join(",");
          await DictAPI.deleteDictByIds(dictIds);
          createdDictIds.push(...dictPageResult.list.map((d) => d.id!));
        }
      } catch (e) {}
    });

    test("强制删除：force=true 时级联删除关联的字典数据", async () => {
      const dictTypeForm = createDictTypeForm();
      await DictAPI.addDictType(dictTypeForm);
      const dictTypePageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: dictTypeForm.code! })
      );
      const dictType = dictTypePageResult.list.find((d) => d.code === dictTypeForm.code);
      expect(dictType).toBeDefined();
      const dictTypeId = dictType!.id;

      const dictForm = createDictForm({
        typeCode: dictTypeForm.code || "",
      });
      await DictAPI.addDict(dictForm);

      await DictAPI.deleteDictTypes(dictTypeId!.toString(), true);

      await expectBizError(DictAPI.getDictTypeForm(dictTypeId!), "A0401", "不存在");

      const dictPageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: dictTypeForm.code!, pageSize: 1000 })
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
  });

  describe("GET /api/v1/dict/page - 字典分页列表", () => {
    let testTypeCode: string;

    beforeAll(async () => {
      // getDictPage 后端要求 typeCode 必填，先获取一个存在的 typeCode
      const typePageResult = await DictAPI.getDictTypePage(createDictTypeQuery({ pageSize: 1 }));
      if (typePageResult.list.length === 0) {
        // 没有字典类型则创建一个
        const form = createDictTypeForm();
        await DictAPI.addDictType(form);
        testTypeCode = form.code!;
        const pageResult = await DictAPI.getDictTypePage(
          createDictTypeQuery({ keywords: testTypeCode })
        );
        const dictType = pageResult.list.find((d) => d.code === testTypeCode);
        if (dictType?.id) createdDictTypeIds.push(dictType.id);
      } else {
        testTypeCode = typePageResult.list[0]!.code!;
      }

      // 创建测试字典数据项，确保搜索测试有数据可用
      for (let i = 0; i < 3; i++) {
        const dictForm = createDictForm({ typeCode: testTypeCode });
        await DictAPI.addDict(dictForm);
        // 查询并记录创建的字典ID用于清理
        const dictPage = await DictAPI.getDictPage(
          createDictQuery({ typeCode: testTypeCode, keywords: dictForm.name!, pageSize: 100 })
        );
        const created = dictPage.list.find((d) => d.name === dictForm.name);
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
      expect(firstDict.name).toBeTruthy();
      const searchKeyword = firstDict.name!.substring(0, 1);

      const result = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, keywords: searchKeyword })
      );
      expect(result.list.length).toBeGreaterThan(0);

      result.list.forEach((item) => {
        expect(item.name).toBeTruthy();
        expect(item.name!.toLowerCase()).toContain(searchKeyword.toLowerCase());
      });
    });
  });

  describe("GET /api/v1/dict/{id}/form - 字典数据表单数据", () => {
    test("获取字典数据表单数据并验证数据准确性", async () => {
      // getDictPage 要求 typeCode 必填，先获取一个存在 typeCode 下有字典数据的页
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
      const form = createDictTypeForm();
      await DictAPI.addDictType(form);
      testTypeCode = form.code!;

      const pageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: testTypeCode })
      );
      const dictType = pageResult.list.find((d) => d.code === testTypeCode);
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }
    });

    test("创建字典并验证数据真实持久化", async () => {
      const testRemark = "这是一个测试字典";
      const form = createDictForm({
        typeCode: testTypeCode,
        remark: testRemark,
      });

      // addDict 返回 undefined（后端 Result<Void>，data:null 被 Jackson 省略）
      await DictAPI.addDict(form);

      const pageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, keywords: form.name! })
      );
      const createdDict = pageResult.list.find(
        (dict) => dict.name === form.name && dict.value === form.value
      );
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
      const form = createDictForm({
        typeCode: testTypeCode,
        sort: 2,
        remark: testRemark,
      });

      await DictAPI.addDict(form);

      const pageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, keywords: form.name! })
      );
      const createdDict = pageResult.list.find(
        (dict) => dict.name === form.name && dict.value === form.value
      );
      expect(createdDict).toBeDefined();

      if (createdDict?.id) {
        createdDictIds.push(createdDict.id);
        const formData = await DictAPI.getDictFormData(createdDict.id);
        expect(formData.remark).toBe(testRemark);
      }
    });

    test("创建禁用状态的字典并验证状态值", async () => {
      const form = createDictForm({
        typeCode: testTypeCode,
        sort: 3,
        status: 0,
      });

      await DictAPI.addDict(form);

      const pageResult = await DictAPI.getDictPage(
        createDictQuery({ typeCode: testTypeCode, keywords: form.name! })
      );
      const createdDict = pageResult.list.find(
        (dict) => dict.name === form.name && dict.value === form.value
      );
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
      const form = createDictForm({
        typeCode: uniqueCode("NON_EXIST"),
      });

      await expectBizError(DictAPI.addDict(form), "A0401", ["类型", "typeCode"]);
    });
  });

  describe("PUT /api/v1/dict/{id} - 修改字典", () => {
    let testDictId: number;
    let testTypeCode: string;
    let originalDictForm: DictForm;

    beforeAll(async () => {
      const dictTypeForm = createDictTypeForm();
      await DictAPI.addDictType(dictTypeForm);
      testTypeCode = dictTypeForm.code || "";

      const dictTypePageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: testTypeCode })
      );
      const dictType = dictTypePageResult.list.find((d) => d.code === testTypeCode);
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      const dictForm = createDictForm({ typeCode: testTypeCode });
      await DictAPI.addDict(dictForm);

      const pageResult = await DictAPI.getDictPage(createDictQuery({ typeCode: testTypeCode }));
      const createdDict = pageResult.list.find((dict) => dict.name === dictForm.name);
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
  });

  describe("DELETE /api/v1/dict/{ids} - 删除字典", () => {
    let testDictIds: number[] = [];
    let testTypeCode: string;

    beforeAll(async () => {
      const dictTypeForm = createDictTypeForm();
      await DictAPI.addDictType(dictTypeForm);
      testTypeCode = dictTypeForm.code || "";

      const dictTypePageResult = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: testTypeCode })
      );
      const dictType = dictTypePageResult.list.find((d) => d.code === testTypeCode);
      if (dictType?.id) {
        createdDictTypeIds.push(dictType.id);
      }

      for (let i = 0; i < 3; i++) {
        const form = createDictForm({
          typeCode: testTypeCode,
          sort: i + 1,
        });
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
      // 两种情况均为参数校验错误
      await expectBizError(
        DictAPI.deleteDictByIds("invalid"),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("完整 CRUD 生命周期：字典类型+字典数据联合生命周期", async () => {
      // Create: 创建字典类型
      const dictTypeForm = createDictTypeForm({ remark: "CRUD生命周期测试" });
      await DictAPI.addDictType(dictTypeForm);

      const typePage = await DictAPI.getDictTypePage(
        createDictTypeQuery({ keywords: dictTypeForm.code! })
      );
      const createdType = typePage.list.find((d) => d.code === dictTypeForm.code);
      expect(createdType).toBeDefined();
      const dictTypeId = createdType!.id!;

      // Read: 验证字典类型字段
      const typeFormData = await DictAPI.getDictTypeForm(dictTypeId);
      expect(typeFormData.name).toBe(dictTypeForm.name);
      expect(typeFormData.code).toBe(dictTypeForm.code);
      expect(typeFormData.remark).toBe("CRUD生命周期测试");

      // Create: 创建字典数据
      const dictForm = createDictForm({ typeCode: dictTypeForm.code!, sort: 1 });
      await DictAPI.addDict(dictForm);

      const dictPage = await DictAPI.getDictPage(
        createDictQuery({ typeCode: dictTypeForm.code!, keywords: dictForm.name! })
      );
      const createdDict = dictPage.list.find(
        (d) => d.name === dictForm.name && d.value === dictForm.value
      );
      expect(createdDict).toBeDefined();
      const dictId = createdDict!.id!;

      // Read: 验证字典数据字段
      const dictFormData = await DictAPI.getDictFormData(dictId);
      expect(dictFormData.name).toBe(dictForm.name);
      expect(dictFormData.value).toBe(dictForm.value);
      expect(dictFormData.typeCode).toBe(dictTypeForm.code);

      // Update: 更新字典数据名称
      const newDictName = `CRUD更新_${Date.now()}`;
      await DictAPI.updateDict(dictId, { ...dictFormData, name: newDictName });

      // Read: 验证更新已生效
      const updatedDict = await DictAPI.getDictFormData(dictId);
      expect(updatedDict.name).toBe(newDictName);

      // Delete: 删除字典数据
      await DictAPI.deleteDictByIds(dictId.toString());

      // Verify: 验证字典数据已不存在
      await expectBizError(DictAPI.getDictFormData(dictId), "A0401", "不存在");

      // Delete: 删除字典类型
      await DictAPI.deleteDictTypes(dictTypeId.toString());

      // Verify: 验证字典类型已不存在
      await expectBizError(DictAPI.getDictTypeForm(dictTypeId), "A0401", "不存在");
    });
  });
});
