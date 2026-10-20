import { RoleForm, RoleQuery } from "@/api/role/model";
import { uniqueName, uniqueCode, pageQuery } from "./common";

export function createRoleForm(overrides: Partial<RoleForm> = {}): RoleForm {
  const code = overrides.code ?? uniqueCode("TEST_ROLE");
  return {
    code,
    name: uniqueName("测试角色"),
    status: 1,
    sort: 100,
    dataScope: 1,
    ...overrides,
  };
}

export function createRoleQuery(overrides: Partial<RoleQuery> = {}): RoleQuery {
  return pageQuery<RoleQuery>({
    ...overrides,
  });
}
