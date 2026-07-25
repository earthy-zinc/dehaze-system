import { DeptForm, DeptQuery } from "@/api/dept/model";
import { uniqueName } from "./common";

/**
 * 创建部门表单数据
 * @param overrides 覆盖默认值的字段
 */
export function createDeptForm(overrides: Partial<DeptForm> = {}): DeptForm {
  return {
    name: uniqueName("测试部门"),
    parentId: 0,
    sort: 100,
    status: 1,
    ...overrides,
  };
}

/**
 * 创建部门查询参数
 * @param overrides 覆盖默认值的字段
 */
export function createDeptQuery(overrides: Partial<DeptQuery> = {}): DeptQuery {
  return {
    ...overrides,
  };
}
