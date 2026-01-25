import { Algorithm, AlgorithmQuery } from "@/api/algorithm/model";
import { uniqueName } from "./common";

/**
 * 创建算法表单数据（用于新增和修改）
 * 注意：不包含 path 字段，因为后端有 @FileExists 验证
 * @param overrides 覆盖默认值的字段
 */
export function createAlgorithmForm(overrides: Partial<Algorithm> = {}): Partial<Algorithm> {
  const baseForm: Partial<Algorithm> = {
    parentId: 0,
    name: uniqueName("测试算法"),
    type: "TEST",
    description: "测试算法描述",
    status: 1,
  };
  return {
    ...baseForm,
    ...overrides,
  };
}

/**
 * 创建算法查询参数
 * @param overrides 覆盖默认值的字段
 */
export function createAlgorithmQuery(overrides: Partial<AlgorithmQuery> = {}): AlgorithmQuery {
  return {
    ...overrides,
  };
}
