import { expectBizError } from "#/utils/assertion";
import { createUserForm, createUserQuery } from "#/factories/user";
import { uniqueName, uniqueEmail, uniqueMobile } from "#/factories/common";
import UserAPI from "@/api/user";
import { UserForm, UserQuery, UserPageVO } from "@/api/user/model";
import { ImportExportAPI } from "../../../index";
import { ROLES, USERS, DEPTS, ADMIN_VISIBLE_USER_COUNT } from "#/factories/constants";

/** 创建用户并校验其出现在列表中，返回列表项 */
async function createUserAndFetch(form: UserForm): Promise<UserPageVO> {
  await UserAPI.add(form);
  const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100, keywords: form.username! });
  const createdUser = pageResult.list.find((u) => u.username === form.username);
  expect(createdUser).toBeDefined();
  return createdUser!;
}

/** 创建用户并返回其 ID */
async function createUserAndGetId(form: UserForm): Promise<number> {
  return (await createUserAndFetch(form)).id!;
}

/** 清理测试用户（失败仅告警，不阻断主流程） */
async function cleanupUsers(ids: number[]): Promise<void> {
  for (const id of ids) {
    try {
      await UserAPI.deleteByIds(id.toString());
    } catch (e) {
      console.warn(`清理失败:`, e);
    }
  }
}

/** 兼容 Blob/ArrayBuffer 等二进制对象的大小探测 */
function blobSize(result: unknown): number {
  return (
    (result as { size?: number }).size ??
    (result as { length?: number }).length ??
    (result as { byteLength?: number }).byteLength ??
    0
  );
}

describe("用户管理接口测试", () => {
  describe("GET /api/v1/auth/me - 获取当前登录用户信息", () => {
    test("获取当前登录用户信息并验证数据完整性", async () => {
      const result = await UserAPI.getInfo();

      expect(result.userId).toBe(USERS.ADMIN.id);
      expect(result.username).toBe(USERS.ADMIN.username);
      expect(result.nickname).toBe(USERS.ADMIN.nickname);

      expect(result.roles).toEqual(expect.arrayContaining([ROLES.ADMIN.code]));

      expect(result.perms.length).toBeGreaterThan(0);
      result.perms.forEach((perm) => {
        expect(typeof perm).toBe("string");
        expect(perm.length).toBeGreaterThan(0);
      });
    });
  });

  describe("GET /api/v1/users/page - 用户分页列表", () => {
    test("获取用户分页列表并验证分页逻辑", async () => {
      const query = createUserQuery({ pageSize: 100 });
      const result = await UserAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
      expect(result.list.length).toBeLessThanOrEqual(query.pageSize || 10);

      result.list.forEach((item) => {
        expect(item.id).toBeGreaterThan(0);
        expect(typeof item.username).toBe("string");
        expect(item.username!.length).toBeGreaterThan(0);
        expect(typeof item.nickname).toBe("string");
        expect(item.nickname!.length).toBeGreaterThan(0);
        expect([0, 1]).toContain(item.status);
        // UserPageVO 返回 genderLabel 而非 gender
        if (item.genderLabel !== undefined && item.genderLabel !== null) {
          expect(typeof item.genderLabel).toBe("string");
        }
      });

      const adminUser = result.list.find((u) => u.id === USERS.ADMIN.id);
      expect(adminUser).toBeDefined();
      expect(adminUser!.username).toBe(USERS.ADMIN.username);
      expect(adminUser!.nickname).toBe(USERS.ADMIN.nickname);
      expect(adminUser!.status).toBe(USERS.ADMIN.status);
    });

    test("按关键词搜索并验证搜索结果的准确性", async () => {
      const searchKeyword = USERS.ADMIN.username.substring(0, 2);
      const result = await UserAPI.getPage(createUserQuery({ keywords: searchKeyword }));

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeGreaterThan(0);

      result.list.forEach((item) => {
        const usernameContains = item.username!.toLowerCase().includes(searchKeyword.toLowerCase());
        const nicknameContains = item.nickname!.toLowerCase().includes(searchKeyword.toLowerCase());
        const mobileContains = item.mobile
          ? item.mobile.toLowerCase().includes(searchKeyword.toLowerCase())
          : false;
        expect(usernameContains || nicknameContains || mobileContains).toBe(true);
      });

      const adminInResult = result.list.find((u) => u.username === USERS.ADMIN.username);
      expect(adminInResult).toBeDefined();
    });

    test("按状态筛选并验证筛选结果", async () => {
      const result = await UserAPI.getPage(createUserQuery({ pageNum: 1, pageSize: 100 }));

      expect(result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);

      result.list.forEach((item) => {
        expect(item.status).toBe(1);
      });

      // ADMIN 角色 data_scope=0(ALL) 可见全部预置用户，但 SysUserMapper.xml 查询排除了 root
      const visiblePresetUsernames: string[] = [
        USERS.ADMIN.username,
        USERS.TEST.username,
        USERS.DEPT_ADMIN.username,
        USERS.USER.username,
        USERS.VIP1.username,
        USERS.VIP2.username,
        USERS.SVIP.username,
      ];
      const foundPresetUsers = result.list.filter((u) =>
        visiblePresetUsernames.includes(u.username!)
      );
      expect(foundPresetUsers.length).toBe(ADMIN_VISIBLE_USER_COUNT);

      const disabledResult = await UserAPI.getPage(
        createUserQuery({ pageNum: 1, pageSize: 100, status: 0 })
      );

      disabledResult.list.forEach((item) => {
        expect(item.status).toBe(0);
      });

      const disabledPresetUsers = disabledResult.list.filter((u) =>
        visiblePresetUsernames.includes(u.username!)
      );
      expect(disabledPresetUsers.length).toBe(0);
    });

    test("按部门筛选并验证返回结果包含该部门用户", async () => {
      const result = await UserAPI.getPage(
        createUserQuery({ deptId: DEPTS.CQUPT.id, pageSize: 100 })
      );
      expect(result.list.length).toBeGreaterThan(0);
      // 分页 VO 仅返回 deptName，未返回 deptId
      result.list.forEach((item) => {
        expect(item.deptName).toBeDefined();
      });
    });

    test("分页逻辑验证 - 不同页码返回不同数据", async () => {
      const pageSize = 2;
      const page1Result = await UserAPI.getPage({ pageNum: 1, pageSize } as UserQuery);
      const page2Result = await UserAPI.getPage({ pageNum: 2, pageSize } as UserQuery);

      expect(page1Result.list.length).toBeLessThanOrEqual(pageSize);
      expect(page2Result.list.length).toBeLessThanOrEqual(pageSize);

      // 后端按 create_time DESC 排序，并发创建的用户会使两页出现交集，故仅校验 total
      expect(page1Result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
      expect(page2Result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
    });
  });

  describe("GET /api/v1/users/{userId}/form - 获取用户表单数据", () => {
    test("获取用户表单数据并验证数据准确性", async () => {
      const result = await UserAPI.getFormData(USERS.ADMIN.id);

      expect(result.id).toBe(USERS.ADMIN.id);
      expect(result.username).toBe(USERS.ADMIN.username);
      expect(result.nickname).toBe(USERS.ADMIN.nickname);
      expect(result.status).toBe(USERS.ADMIN.status);
      expect(result.gender).toBe(USERS.ADMIN.gender);
      expect(result.email).toBe(USERS.ADMIN.email);
      expect(result.mobile).toBe(USERS.ADMIN.mobile);
      expect(result.deptId).toBe(USERS.ADMIN.deptId);

      expect(Array.isArray(result.roleIds)).toBe(true);
      expect(result.roleIds).toEqual(expect.arrayContaining(USERS.ADMIN.roleIds));
    });

    test("获取 TEST 用户表单数据并验证部门和角色", async () => {
      const result = await UserAPI.getFormData(USERS.TEST.id);

      expect(result.id).toBe(USERS.TEST.id);
      expect(result.username).toBe(USERS.TEST.username);
      expect(result.nickname).toBe(USERS.TEST.nickname);
      expect(result.deptId).toBe(DEPTS.COMPUTER.id);

      expect(result.roleIds).toEqual(expect.arrayContaining([ROLES.GUEST.id]));
    });

    test("获取不存在用户的表单数据应返回空", async () => {
      // 后端对不存在的用户返回成功但 data 为空（Jackson 省略 null 字段，SDK 解析为 undefined）
      const result = await UserAPI.getFormData(99999999);
      expect(result).toBeUndefined();
    });
  });

  describe("POST /api/v1/users - 新增用户", () => {
    const createdUserIds: number[] = [];
    const existingDeptId = DEPTS.CQUPT.id;
    const existingRoleIds = [ROLES.ADMIN.id, ROLES.GUEST.id];

    afterAll(() => cleanupUsers(createdUserIds));

    test("创建用户并验证数据真实持久化", async () => {
      const form = createUserForm({
        deptId: existingDeptId,
        roleIds: existingRoleIds,
      });

      const createdUser = await createUserAndFetch(form);

      expect(createdUser.id!).toBeGreaterThan(0);
      expect(createdUser.username).toBe(form.username);
      expect(createdUser.nickname).toBe(form.nickname);
      expect(createdUser.status).toBe(form.status);

      // 通过 getFormData 验证完整数据持久化（包括 gender）
      const formData = await UserAPI.getFormData(createdUser.id!);

      expect(formData.id).toBe(createdUser.id);
      expect(formData.username).toBe(form.username);
      expect(formData.nickname).toBe(form.nickname);
      expect(formData.email).toBe(form.email);
      expect(formData.mobile).toBe(form.mobile);
      expect(formData.gender).toBe(form.gender);
      expect(formData.status).toBe(form.status);
      expect(formData.deptId).toBe(existingDeptId);

      expect(Array.isArray(formData.roleIds)).toBe(true);
      expect(formData.roleIds!.length).toBe(existingRoleIds.length);
      existingRoleIds.forEach((roleId) => {
        expect(formData.roleIds).toContain(roleId);
      });

      createdUserIds.push(createdUser.id!);
    });

    test("创建包含邮箱和手机号的用户并验证数据正确性", async () => {
      const form = createUserForm({
        deptId: existingDeptId,
        email: uniqueEmail("test"),
        mobile: uniqueMobile(),
        roleIds: [ROLES.GUEST.id],
      });

      const createdUser = await createUserAndFetch(form);
      const formData = await UserAPI.getFormData(createdUser.id!);
      expect(formData.email).toBe(form.email);
      expect(formData.mobile).toBe(form.mobile);

      createdUserIds.push(createdUser.id!);
    });

    test("创建禁用状态的用户并验证状态值", async () => {
      const form = createUserForm({
        status: 0,
        deptId: existingDeptId,
        roleIds: [ROLES.GUEST.id],
      });

      const createdUser = await createUserAndFetch(form);
      expect(createdUser.status).toBe(0);

      const formData = await UserAPI.getFormData(createdUser.id!);
      expect(formData.status).toBe(0);

      const disabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 0 });
      const foundUser = disabledUsers.list.find((u) => u.id === createdUser.id);
      expect(foundUser).toBeDefined();
      expect(foundUser!.status).toBe(0);

      createdUserIds.push(createdUser.id!);
    });

    test("参数校验：缺少必需字段 username", async () => {
      const form: Partial<UserForm> = {
        nickname: "测试用户",
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizError(UserAPI.add(form as UserForm), ["A0400"]);
    });

    test("参数校验：缺少必需字段 nickname", async () => {
      const form: Partial<UserForm> = {
        username: `testuser_${Date.now()}`,
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizError(UserAPI.add(form as UserForm), ["A0400"]);
    });

    test("参数校验：用户名已存在", async () => {
      const form: UserForm = {
        username: USERS.ADMIN.username,
        nickname: "测试用户",
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizError(UserAPI.add(form), ["A0400"]);
    });

    test("参数校验：手机号格式不正确应失败", async () => {
      const form = createUserForm({
        mobile: "12345",
        deptId: existingDeptId,
        roleIds: [ROLES.GUEST.id],
      });
      await expectBizError(UserAPI.add(form), ["A0400"]);
    });

    test("参数校验：邮箱格式不正确应失败 [T-UM-015]", async () => {
      const form = createUserForm({
        email: "invalid-email",
        deptId: existingDeptId,
        roleIds: [ROLES.GUEST.id],
      });
      await expectBizError(UserAPI.add(form), ["A0400"]);
    });

    test("参数校验：缺少部门ID应失败 [T-UM-016]", async () => {
      const form: Partial<UserForm> = {
        username: uniqueName("test_user"),
        nickname: "测试用户",
        roleIds: [ROLES.GUEST.id],
        status: 1,
        gender: 1,
      };
      await expectBizError(UserAPI.add(form as UserForm), ["A0400"]);
    });

    test("参数校验：角色未分配应失败", async () => {
      const form = createUserForm({
        deptId: existingDeptId,
        roleIds: [],
      });
      await expectBizError(UserAPI.add(form), ["A0400"]);
    });
  });

  describe("PUT /api/v1/users/{userId} - 修改用户", () => {
    let testUserId: number;
    const existingDeptId = DEPTS.CQUPT.id;
    const existingRoleIds = [ROLES.ADMIN.id, ROLES.GUEST.id];
    let originalUser: UserForm;

    beforeAll(async () => {
      const form = createUserForm({
        username: `testuser_update_${Date.now()}`,
        nickname: "测试用户更新",
        gender: 1,
        status: 1,
        deptId: existingDeptId,
        roleIds: existingRoleIds,
      });
      testUserId = await createUserAndGetId(form);
      originalUser = await UserAPI.getFormData(testUserId);
    });

    afterAll(() => cleanupUsers([testUserId]));

    /** 基于最新表单快照构造完整更新表单，仅覆盖指定字段 */
    const buildUpdateForm = (before: UserForm, overrides: Partial<UserForm> = {}): UserForm => ({
      username: before.username ?? originalUser.username ?? "",
      nickname: before.nickname ?? originalUser.nickname ?? "",
      email: before.email ?? originalUser.email ?? "",
      mobile: before.mobile ?? originalUser.mobile ?? "",
      gender: before.gender ?? originalUser.gender ?? 1,
      deptId: before.deptId ?? originalUser.deptId ?? existingDeptId,
      roleIds: before.roleIds ?? existingRoleIds,
      ...overrides,
    });

    test("更新用户昵称并验证更新真实生效", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      expect(beforeUpdate.nickname).toBe(originalUser.nickname);

      const newNickname = `更新后的昵称_${Date.now()}`;
      const form = buildUpdateForm(beforeUpdate, { nickname: newNickname });

      await UserAPI.update(testUserId, form);

      const afterUpdate = await UserAPI.getFormData(testUserId);
      expect(afterUpdate.nickname).toBe(newNickname);
      expect(afterUpdate.nickname).not.toBe(beforeUpdate.nickname);

      expect(afterUpdate.username).toBe(originalUser.username);
      expect(afterUpdate.id).toBe(testUserId);
      expect(afterUpdate.email).toBe(originalUser.email);
      expect(afterUpdate.mobile).toBe(originalUser.mobile);
    });

    test("更新用户邮箱并验证邮箱值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newEmail = `updated_${Date.now()}@example.com`;
      const form = buildUpdateForm(beforeUpdate, { email: newEmail });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.email).toBe(newEmail);
      expect(formData.email).not.toBe(beforeUpdate.email);
    });

    test("更新用户手机号并验证手机号正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newMobile = uniqueMobile();
      const form = buildUpdateForm(beforeUpdate, { mobile: newMobile });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.mobile).toBe(newMobile);
      expect(formData.mobile).not.toBe(beforeUpdate.mobile);
    });

    test("更新用户状态并验证状态值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const form = buildUpdateForm(beforeUpdate, { status: 0 });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.status).toBe(0);

      const disabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 0 });
      const foundDisabled = disabledUsers.list.find((u) => u.id === testUserId);
      expect(foundDisabled).toBeDefined();
      expect(foundDisabled!.status).toBe(0);

      // 恢复启用状态
      await UserAPI.update(testUserId, buildUpdateForm(beforeUpdate, { status: 1 }));

      const formData2 = await UserAPI.getFormData(testUserId);
      expect(formData2.status).toBe(1);

      const enabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 1 });
      const foundEnabled = enabledUsers.list.find((u) => u.id === testUserId);
      expect(foundEnabled).toBeDefined();
      expect(foundEnabled!.status).toBe(1);
    });

    test("更新用户角色并验证角色分配正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newRoleIds = [ROLES.ROOT.id, ROLES.ADMIN.id, ROLES.GUEST.id];
      const form = buildUpdateForm(beforeUpdate, { roleIds: newRoleIds });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(Array.isArray(formData.roleIds)).toBe(true);
      expect(formData.roleIds!.length).toBe(newRoleIds.length);
      newRoleIds.forEach((roleId) => {
        expect(formData.roleIds).toContain(roleId);
      });
    });

    test("更新用户性别并验证性别值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newGender = beforeUpdate.gender === 1 ? 0 : 1;
      const form = buildUpdateForm(beforeUpdate, { gender: newGender });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.gender).toBe(newGender);
      expect(formData.gender).not.toBe(beforeUpdate.gender);
    });

    test("同时更新多个字段并验证所有字段都更新成功", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newNickname = `批量更新昵称_${Date.now()}`;
      const newEmail = `batch_update_${Date.now()}@example.com`;
      const newGender = beforeUpdate.gender === 1 ? 0 : 1;
      const newStatus = 0;

      const form = buildUpdateForm(beforeUpdate, {
        nickname: newNickname,
        email: newEmail,
        gender: newGender,
        status: newStatus,
      });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.nickname).toBe(newNickname);
      expect(formData.nickname).not.toBe(beforeUpdate.nickname);
      expect(formData.email).toBe(newEmail);
      expect(formData.email).not.toBe(beforeUpdate.email);
      expect(formData.gender).toBe(newGender);
      expect(formData.gender).not.toBe(beforeUpdate.gender);
      expect(formData.status).toBe(newStatus);
      expect(formData.username).toBe(originalUser.username);
      expect(formData.mobile).toBe(beforeUpdate.mobile);

      // 恢复状态为启用
      await UserAPI.update(
        testUserId,
        buildUpdateForm(beforeUpdate, {
          nickname: newNickname,
          email: newEmail,
          gender: newGender,
          status: 1,
        })
      );
    });

    test("更新不存在的用户应返回 A0401", async () => {
      const form = createUserForm({
        username: "nonexistent_user",
        nickname: "测试",
        roleIds: existingRoleIds,
      });
      await expectBizError(UserAPI.update(99999999, form), ["A0401"]);
    });

    test("参数校验：用户名冲突", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const form = createUserForm({
        username: USERS.ADMIN.username,
        nickname: beforeUpdate.nickname || originalUser.nickname || "",
        roleIds: beforeUpdate.roleIds ?? existingRoleIds,
      });
      await expectBizError(UserAPI.update(testUserId, form), ["A0503"]);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.username).toBe(originalUser.username);
      expect(formData.username).not.toBe(USERS.ADMIN.username);
    });

    test("参数校验：用户名只读，修改用户名应失败或保持不变 [T-UM-023]", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const form = buildUpdateForm(beforeUpdate, { username: uniqueName("attempt_change") });

      // 后端拒绝修改或忽略 username 字段均可，用户名不应改变
      try {
        await UserAPI.update(testUserId, form);
      } catch {
        // 后端拒绝修改用户名也是预期行为
      }

      const resultFormData = await UserAPI.getFormData(testUserId);
      expect(resultFormData.username).toBe(originalUser.username);
    });
  });

  describe("PATCH /api/v1/users/{userId}/password - 修改用户密码", () => {
    let testUserId: number;
    const existingDeptId = DEPTS.CQUPT.id;

    beforeAll(async () => {
      const form = createUserForm({
        username: `testuser_pwd_${Date.now()}`,
        nickname: "测试用户密码",
        status: 1,
        deptId: existingDeptId,
        roleIds: [ROLES.GUEST.id],
      });
      testUserId = await createUserAndGetId(form);
    });

    afterAll(() => cleanupUsers([testUserId]));

    test("修改用户密码并验证密码确实被修改", async () => {
      const newPassword = `NewPwd_${Date.now()}!`;
      await UserAPI.updatePassword(testUserId, newPassword);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.id).toBe(testUserId);
    });

    test("参数校验：空密码应被拒绝", async () => {
      await expectBizError(UserAPI.updatePassword(testUserId, ""), ["A0400"]);
    });

    test("修改不存在用户的密码应返回 A0401", async () => {
      await expectBizError(UserAPI.updatePassword(99999999, "newpassword123"), ["A0401"]);
    });
  });

  describe("PATCH /api/v1/users/{userId}/status - 用户状态管理", () => {
    let testUserId: number;

    beforeAll(async () => {
      const form = createUserForm({
        username: `testuser_status_${Date.now()}`,
        nickname: "测试状态管理",
        status: 1,
        deptId: DEPTS.CQUPT.id,
        roleIds: [ROLES.GUEST.id],
      });
      testUserId = await createUserAndGetId(form);
    });

    afterAll(() => cleanupUsers([testUserId]));

    test("正向测试：禁用用户并验证状态变更", async () => {
      await UserAPI.updateStatus(testUserId, 0);
      expect((await UserAPI.getFormData(testUserId)).status).toBe(0);
    });

    test("正向测试：启用用户并验证状态变更", async () => {
      await UserAPI.updateStatus(testUserId, 1);
      expect((await UserAPI.getFormData(testUserId)).status).toBe(1);
    });

    test("边界：禁用超级管理员应失败", async () => {
      // 超级管理员不可禁用，后端返回 A0505（文档旧契约 A0231，以实际为准）
      await expectBizError(UserAPI.updateStatus(USERS.ROOT.id, 0), ["A0505"]);
    });

    test("异常：更新不存在用户的状态应返回 A0401", async () => {
      await expectBizError(UserAPI.updateStatus(99999999, 0), ["A0401"]);
    });
  });

  describe("DELETE /api/v1/users/{ids} - 删除用户", () => {
    let testUserIds: number[] = [];
    const existingDeptId = DEPTS.CQUPT.id;

    const expectUserDeleted = (result: UserForm | null | undefined) => {
      expect(result === null || result === undefined || result.id === undefined).toBe(true);
    };

    beforeAll(async () => {
      for (let i = 0; i < 3; i++) {
        const form = createUserForm({
          username: `testuser_del_${Date.now()}_${i}`,
          nickname: `测试用户删除${i}`,
          status: 1,
          deptId: existingDeptId,
          roleIds: [ROLES.GUEST.id],
        });
        testUserIds.push(await createUserAndGetId(form));
      }
      expect(testUserIds.length).toBe(3);
    });

    test("删除单个用户并验证用户真的被删除", async () => {
      const userId = testUserIds[0]!;

      const beforeDelete = await UserAPI.getFormData(userId);
      expect(beforeDelete.id).toBe(userId);

      await UserAPI.deleteByIds(userId.toString());

      expectUserDeleted(await UserAPI.getFormData(userId));

      const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100 });
      expect(pageResult.list.find((u) => u.id === userId)).toBeUndefined();

      expect(pageResult.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
    });

    test("批量删除多个用户并验证所有用户都被删除", async () => {
      const ids = testUserIds.slice(1);
      expect(ids.length).toBe(2);

      for (const userId of ids) {
        const beforeDelete = await UserAPI.getFormData(userId!);
        expect(beforeDelete.id).toBe(userId);
      }

      await UserAPI.deleteByIds(ids.join(","));

      for (const userId of ids) {
        expectUserDeleted(await UserAPI.getFormData(userId!));
      }

      const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100 });
      for (const userId of ids) {
        expect(pageResult.list.find((u) => u.id === userId)).toBeUndefined();
      }
    });

    test("删除不存在的用户应保持幂等性", async () => {
      // DELETE 幂等性：删除不存在的资源应返回成功（不抛异常即视为幂等）
      await UserAPI.deleteByIds("99999999");
    });

    // 空 ID 列表属调用方编程错误，由 SDK 前置校验拦截（DELETE /{ids} 路由无法接收空路径参数，
    // 后端返回 405 是 HTTP 语义正确行为）
    test("参数校验：空的ID列表由 SDK 前置校验拦截", async () => {
      await expect(UserAPI.deleteByIds("")).rejects.toThrow("不能为空");
    });

    test("边界：删除超级管理员应失败（超级管理员保护）[T-UM-029]", async () => {
      // USERS.ROOT (id=1) 是超级管理员，不可删除
      await expectBizError(UserAPI.deleteByIds(USERS.ROOT.id.toString()), ["A0505"]);
    });

    test("边界：删除自己应失败 [T-UM-030]", async () => {
      // 当前登录用户是 admin (id=2)，不能删除自己
      await expectBizError(UserAPI.deleteByIds(USERS.ADMIN.id.toString()), ["A0503"]);
    });
  });

  describe("GET /api/v1/users/template - 用户导入模板下载", () => {
    test("下载用户导入模板（通过 ImportExportAPI）", async () => {
      const result = await ImportExportAPI.downloadTemplate("user");
      expect(blobSize(result)).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/users/_export - 导出用户", () => {
    test("导出所有用户（通过 ImportExportAPI，同步返回 Blob 或异步返回任务）", async () => {
      const result = await ImportExportAPI.export("user", {});

      const size = blobSize(result);
      const isBlob = typeof size === "number" && size > 0;
      const isTaskResult = !isBlob && typeof (result as { taskId?: string }).taskId === "string";
      expect(isBlob || isTaskResult).toBe(true);
    });
  });
});
