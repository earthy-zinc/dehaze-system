import { DeptForm, DeptQuery } from "@/api/dept/model";
import { uniqueName } from "./common";
import { TestCleanupRegistry } from "#/utils/cleanup";
import DeptAPI from "@/api/dept";

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

/**
 * 创建多级部门链，返回各级部门 ID（按层级升序）
 * @param rootParentId 根父部门 ID（链顶挂载在哪个部门下）
 * @param levels 要创建的层级数（不包含 rootParent，即新建 levels 个部门）
 * @param cleanup 清理注册表（用于注册创建的部门 ID）
 * @returns 各级部门 ID 数组，索引 0 为第一级新建部门
 */
export async function createDeptChain(
  rootParentId: number,
  levels: number,
  cleanup?: TestCleanupRegistry
): Promise<number[]> {
  const ids: number[] = [];

  let currentParentId = rootParentId;
  for (let i = 0; i < levels; i++) {
    const form = createDeptForm({ parentId: currentParentId });
    const deptId = (await DeptAPI.add(form)) as number;
    ids.push(deptId);
    if (cleanup) {
      cleanup.register(async () => {
        try {
          await DeptAPI.deleteByIds(deptId.toString());
        } catch {
          /* 忽略 */
        }
      });
    }
    currentParentId = deptId;
  }

  return ids;
}
