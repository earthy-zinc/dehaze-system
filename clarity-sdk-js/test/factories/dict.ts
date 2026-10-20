import { DictForm, DictQuery, DictTypeForm, DictTypeQuery } from "@/api/dict/model";
import { uniqueName, uniqueCode, pageQuery } from "./common";

export function createDictTypeForm(overrides: Partial<DictTypeForm> = {}): DictTypeForm {
  const code = overrides.code ?? uniqueCode("TEST_TYPE");
  return {
    code,
    name: uniqueName("测试字典类型"),
    status: 1,
    ...overrides,
  };
}

export function createDictTypeQuery(overrides: Partial<DictTypeQuery> = {}): DictTypeQuery {
  return pageQuery<DictTypeQuery>({
    ...overrides,
  });
}

export function createDictForm(overrides: Partial<DictForm> = {}): DictForm {
  return {
    name: uniqueName("测试字典"),
    // 使用时间戳后6位+计数器，确保同一毫秒内创建的多个字典 value 也唯一，
    // 避免同类型下 value 冲突触发 A0501
    value: `${Date.now().toString().slice(-6)}${(Math.random() * 1000).toFixed(0).padStart(3, "0")}`,
    typeCode: overrides.typeCode ?? uniqueCode("TEST_TYPE"),
    sort: 1,
    status: 1,
    ...overrides,
  };
}

export function createDictQuery(overrides: Partial<DictQuery> = {}): DictQuery {
  return pageQuery<DictQuery>({
    ...overrides,
  });
}
