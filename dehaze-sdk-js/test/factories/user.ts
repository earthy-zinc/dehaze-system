import { UserForm, UserQuery } from "@/api/user/model";
import { uniqueName, uniqueEmail, uniqueMobile, pageQuery } from "./common";

export function createUserForm(overrides: Partial<UserForm> = {}): UserForm {
  const username = overrides.username ?? uniqueName("test_user");
  return {
    username,
    nickname: uniqueName("测试用户"),
    email: uniqueEmail(username),
    mobile: uniqueMobile(),
    gender: 1,
    status: 1,
    deptId: 1,
    roleIds: [1],
    ...overrides,
  };
}

export function createUserQuery(overrides: Partial<UserQuery> = {}): UserQuery {
  return pageQuery<UserQuery>({
    status: 1,
    ...overrides,
  });
}
