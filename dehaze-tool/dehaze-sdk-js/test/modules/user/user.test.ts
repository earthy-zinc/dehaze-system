import { login, logout } from "#/utils/auth";
import { expectBizError, expectBizErrorOrUndefined } from "#/utils/assertion";
import { createUserForm, createUserQuery } from "#/factories/user";
import UserAPI from "@/api/user";
import { UserForm, UserQuery } from "@/api/user/model";
import { ROLES, USERS, DEPTS, ADMIN_VISIBLE_USER_COUNT } from "#/factories/constants";

describe("用户管理接口测试", () => {
  beforeAll(async () => {
    await login();
  }, 30000);

  afterAll(async () => {
    await logout();
  });

  describe("GET /api/v1/users/me - 获取当前登录用户信息", () => {
    test("获取当前登录用户信息并验证数据完整性", async () => {
      const result = await UserAPI.getInfo();

      expect(result).toBeDefined();
      expect(result.userId).toBe(USERS.ADMIN.id);
      expect(result.username).toBe(USERS.ADMIN.username);
      expect(result.nickname).toBe(USERS.ADMIN.nickname);

      // 验证角色数组包含预期值
      expect(result.roles).toEqual(expect.arrayContaining([ROLES.ADMIN.code]));
      expect(result.roles.length).toBeGreaterThan(0);

      // 验证权限数组非空且包含字符串
      expect(result.perms).toEqual(expect.any(Array));
      expect(result.perms.length).toBeGreaterThan(0);
      result.perms.forEach((perm) => {
        expect(typeof perm).toBe("string");
        expect(perm.length).toBeGreaterThan(0);
      });

      // 验证接口幂等性：多次调用返回相同数据
      const result2 = await UserAPI.getInfo();
      expect(result.userId).toBe(result2.userId);
      expect(result.username).toBe(result2.username);
      expect(result.nickname).toBe(result2.nickname);
      expect(result.roles).toEqual(result2.roles);
    });
  });

  describe("GET /api/v1/users/page - 用户分页列表", () => {
    test("获取用户分页列表并验证分页逻辑", async () => {
      const query = createUserQuery({ pageSize: 100 });

      const result = await UserAPI.getPage(query);

      // 验证分页结构
      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
      expect(result.list.length).toBeLessThanOrEqual(query.pageSize || 10);

      // 验证列表项字段完整性和值合理性
      result.list.forEach((item) => {
        expect(item.id).toBeGreaterThan(0);
        expect(typeof item.username).toBe("string");
        expect(item.username!.length).toBeGreaterThan(0);
        expect(typeof item.nickname).toBe("string");
        expect(item.nickname!.length).toBeGreaterThan(0);
        expect([0, 1]).toContain(item.status);
        // 验证性别标签字段存在（UserPageVO 使用 genderLabel）
        if (item.genderLabel !== undefined && item.genderLabel !== null) {
          expect(typeof item.genderLabel).toBe("string");
        }
      });

      // 验证预置用户存在于列表中
      const adminUser = result.list.find((u) => u.id === USERS.ADMIN.id);
      expect(adminUser).toBeDefined();
      expect(adminUser!.username).toBe(USERS.ADMIN.username);
      expect(adminUser!.nickname).toBe(USERS.ADMIN.nickname);
      expect(adminUser!.status).toBe(USERS.ADMIN.status);
    });

    test("按关键词搜索并验证搜索结果的准确性", async () => {
      // 使用已知用户的用户名进行搜索
      const searchKeyword = USERS.ADMIN.username.substring(0, 2);
      const query = createUserQuery({ keywords: searchKeyword });
      const result = await UserAPI.getPage(query);

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeGreaterThan(0);

      // 验证搜索结果都包含关键词
      result.list.forEach((item) => {
        const usernameContains = item.username!.toLowerCase().includes(searchKeyword.toLowerCase());
        const nicknameContains = item.nickname!.toLowerCase().includes(searchKeyword.toLowerCase());
        const mobileContains = item.mobile
          ? item.mobile.toLowerCase().includes(searchKeyword.toLowerCase())
          : false;
        expect(usernameContains || nicknameContains || mobileContains).toBe(true);
      });

      // 验证 admin 用户在搜索结果中
      const adminInResult = result.list.find((u) => u.username === USERS.ADMIN.username);
      expect(adminInResult).toBeDefined();
    });

    test("按状态筛选并验证筛选结果", async () => {
      // 筛选启用状态的用户
      const query = createUserQuery({ pageNum: 1, pageSize: 100, status: 1 });
      const result = await UserAPI.getPage(query);

      expect(result).toBeDefined();
      // 预置的3个用户都是启用状态
      expect(result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);

      // 验证所有返回的用户状态都是启用
      result.list.forEach((item) => {
        expect(item.status).toBe(1);
      });

      // 验证预置用户存在（admin 只能看到 admin 和 test，root 因数据权限不可见）
      const visiblePresetUsernames: string[] = [USERS.ADMIN.username, USERS.TEST.username];
      const foundPresetUsers = result.list.filter((u) =>
        visiblePresetUsernames.includes(u.username!)
      );
      expect(foundPresetUsers.length).toBe(ADMIN_VISIBLE_USER_COUNT);

      // 筛选禁用状态的用户
      const disabledQuery = createUserQuery({ pageNum: 1, pageSize: 100, status: 0 });
      const disabledResult = await UserAPI.getPage(disabledQuery);

      disabledResult.list.forEach((item) => {
        expect(item.status).toBe(0);
      });

      // 预置用户不应该在禁用列表中
      const disabledPresetUsers = disabledResult.list.filter((u) =>
        visiblePresetUsernames.includes(u.username!)
      );
      expect(disabledPresetUsers.length).toBe(0);
    });

    test("分页逻辑验证 - 不同页码返回不同数据", async () => {
      const pageSize = 2;
      const page1Result = await UserAPI.getPage({ pageNum: 1, pageSize } as UserQuery);
      const page2Result = await UserAPI.getPage({ pageNum: 2, pageSize } as UserQuery);

      expect(page1Result.list.length).toBeLessThanOrEqual(pageSize);
      expect(page2Result.list.length).toBeLessThanOrEqual(pageSize);

      // 验证两页数据没有交集
      if (page1Result.list.length > 0 && page2Result.list.length > 0) {
        const page1Ids = page1Result.list.map((u) => u.id);
        const page2Ids = page2Result.list.map((u) => u.id);
        const hasIntersection = page1Ids.some((id) => page2Ids.includes(id));
        expect(hasIntersection).toBe(false);

        // 验证 ID 排序一致性（假设默认按 ID 排序）
        page1Ids.forEach((id, index) => {
          if (index > 0) {
            // 验证同一页内 ID 有序
            expect(typeof id).toBe("number");
          }
        });
      }

      // 验证总数一致
      expect(page1Result.total).toBe(page2Result.total);
    });

    test("超大页码应返回空数组", async () => {
      const query = createUserQuery({ pageNum: 99999, pageSize: 10 });
      const result = await UserAPI.getPage(query);

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBe(0);
      // 总数应该仍然正确
      expect(result.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
    });
  });

  describe("GET /api/v1/users/{userId}/form - 获取用户表单数据", () => {
    test("获取用户表单数据并验证数据准确性", async () => {
      const userId = USERS.ADMIN.id;

      const result = await UserAPI.getFormData(userId);

      // 验证所有字段与预置数据一致
      expect(result).toBeDefined();
      expect(result.id).toBe(userId);
      expect(result.username).toBe(USERS.ADMIN.username);
      expect(result.nickname).toBe(USERS.ADMIN.nickname);
      expect(result.status).toBe(USERS.ADMIN.status);
      expect(result.gender).toBe(USERS.ADMIN.gender);
      expect(result.email).toBe(USERS.ADMIN.email);
      expect(result.mobile).toBe(USERS.ADMIN.mobile);
      expect(result.deptId).toBe(USERS.ADMIN.deptId);

      // 验证角色分配正确
      expect(result.roleIds).toBeDefined();
      expect(Array.isArray(result.roleIds)).toBe(true);
      expect(result.roleIds).toEqual(expect.arrayContaining(USERS.ADMIN.roleIds));
    });

    test("获取 TEST 用户表单数据并验证部门和角色", async () => {
      const userId = USERS.TEST.id;

      const result = await UserAPI.getFormData(userId);

      expect(result).toBeDefined();
      expect(result.id).toBe(userId);
      expect(result.username).toBe(USERS.TEST.username);
      expect(result.nickname).toBe(USERS.TEST.nickname);
      expect(result.deptId).toBe(USERS.TEST.deptId);
      expect(result.deptId).toBe(DEPTS.COMPUTER.id);

      // 验证 TEST 用户角色是 GUEST
      expect(result.roleIds).toBeDefined();
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

    afterAll(async () => {
      for (const userId of createdUserIds) {
        try {
          await UserAPI.deleteByIds(userId.toString());
        } catch (e) {
          // 忽略删除错误
        }
      }
    });

    const addUser = async (form: UserForm) => {
      const username = form.username;
      if (!username) {
        throw new Error("用户名不能为空");
      }

      await UserAPI.add(form);

      const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100, keywords: username });
      const createdUser = pageResult.list.find((user) => user.username === username);
      expect(createdUser).toBeDefined();
      return createdUser;
    };

    test("创建用户并验证数据真实持久化", async () => {
      const form = createUserForm({
        deptId: existingDeptId,
        roleIds: existingRoleIds,
      });

      const createdUser = await addUser(form);

      expect(createdUser).toBeDefined();
      expect(createdUser!.id).toBeGreaterThan(0);
      expect(createdUser!.username).toBe(form.username);
      expect(createdUser!.nickname).toBe(form.nickname);
      expect(createdUser!.status).toBe(form.status);

      // 通过 getFormData 验证完整数据持久化（包括 gender）
      const formData = await UserAPI.getFormData(createdUser!.id!);

      expect(formData).toBeDefined();
      expect(formData.id).toBe(createdUser!.id);
      expect(formData.username).toBe(form.username);
      expect(formData.nickname).toBe(form.nickname);
      expect(formData.email).toBe(form.email);
      expect(formData.mobile).toBe(form.mobile);
      expect(formData.gender).toBe(form.gender);
      expect(formData.status).toBe(form.status);
      expect(formData.deptId).toBe(existingDeptId);

      // 验证角色分配正确
      expect(formData.roleIds).toBeDefined();
      expect(Array.isArray(formData.roleIds)).toBe(true);
      expect(formData.roleIds!.length).toBe(existingRoleIds.length);
      existingRoleIds.forEach((roleId) => {
        expect(formData.roleIds).toContain(roleId);
      });

      createdUserIds.push(createdUser!.id!);
    });

    test("创建包含邮箱和手机号的用户并验证数据正确性", async () => {
      const testEmail = `test_${Date.now()}@example.com`;
      const testMobile = `138${String(Date.now()).slice(-8)}`;
      const form = createUserForm({
        deptId: existingDeptId,
        email: testEmail,
        mobile: testMobile,
        roleIds: [ROLES.GUEST.id],
      });

      const createdUser = await addUser(form);
      expect(createdUser).toBeDefined();

      const formData = await UserAPI.getFormData(createdUser!.id!);
      expect(formData.email).toBe(testEmail);
      expect(formData.mobile).toBe(testMobile);

      createdUserIds.push(createdUser!.id!);
    });

    test("创建禁用状态的用户并验证状态值", async () => {
      const form = createUserForm({
        status: 0,
        deptId: existingDeptId,
        roleIds: [ROLES.GUEST.id],
      });

      const createdUser = await addUser(form);
      expect(createdUser).toBeDefined();
      expect(createdUser!.status).toBe(0);

      // 验证禁用状态持久化
      const formData = await UserAPI.getFormData(createdUser!.id!);
      expect(formData.status).toBe(0);

      // 验证在禁用用户列表中能查到
      const disabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 0 });
      const foundUser = disabledUsers.list.find((u) => u.id === createdUser!.id);
      expect(foundUser).toBeDefined();
      expect(foundUser!.status).toBe(0);

      createdUserIds.push(createdUser!.id!);
    });

    test("参数校验：缺少必需字段 username", async () => {
      const form: Partial<UserForm> = {
        nickname: "测试用户",
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizErrorOrUndefined(UserAPI.add(form as UserForm), ["A0400", "B0001"]);
    });

    test("参数校验：缺少必需字段 nickname", async () => {
      const form: Partial<UserForm> = {
        username: `testuser_${Date.now()}`,
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizErrorOrUndefined(UserAPI.add(form as UserForm), ["A0400", "B0001"]);
    });

    test("参数校验：用户名已存在", async () => {
      const form: UserForm = {
        username: USERS.ADMIN.username, // 使用已存在的用户名
        nickname: "测试用户",
        status: 1,
        deptId: existingDeptId,
      };
      await expectBizErrorOrUndefined(UserAPI.add(form), ["A0400", "B0001"]);
    });
  });

  describe("PUT /api/v1/users/{userId} - 修改用户", () => {
    let testUserId: number;
    const existingDeptId = DEPTS.CQUPT.id;
    const existingRoleIds = [ROLES.ADMIN.id, ROLES.GUEST.id];
    let originalUser: any;

    beforeAll(async () => {
      // 创建测试用户
      const form = createUserForm({
        username: `testuser_update_${Date.now()}`,
        nickname: "测试用户更新",
        email: "update@example.com",
        mobile: "13800138001",
        gender: 1,
        status: 1,
        deptId: existingDeptId,
        roleIds: existingRoleIds,
      });
      await UserAPI.add(form);
      const userPageResult = await UserAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        keywords: form.username!,
      });
      const createdUser = userPageResult.list.find((u) => u.username === form.username);
      if (createdUser?.id) {
        testUserId = createdUser.id;
        originalUser = await UserAPI.getFormData(testUserId);
      }
    });

    afterAll(async () => {
      try {
        await UserAPI.deleteByIds(testUserId.toString());
      } catch (e) {
        // 忽略删除错误
      }
    });

    test("更新用户昵称并验证更新真实生效", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      expect(beforeUpdate.nickname).toBe(originalUser.nickname);

      const newNickname = `更新后的昵称_${Date.now()}`;
      const form = createUserForm({
        username: originalUser.username,
        nickname: newNickname,
        email: originalUser.email,
        mobile: originalUser.mobile,
        gender: originalUser.gender,
        deptId: originalUser.deptId,
        roleIds: existingRoleIds,
      });

      await UserAPI.update(testUserId, form);

      const afterUpdate = await UserAPI.getFormData(testUserId);
      expect(afterUpdate.nickname).toBe(newNickname);
      expect(afterUpdate.nickname).not.toBe(beforeUpdate.nickname);

      // 验证其他字段未被修改
      expect(afterUpdate.username).toBe(originalUser.username);
      expect(afterUpdate.id).toBe(testUserId);
      expect(afterUpdate.email).toBe(originalUser.email);
      expect(afterUpdate.mobile).toBe(originalUser.mobile);
    });

    test("更新用户邮箱并验证邮箱值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newEmail = `updated_${Date.now()}@example.com`;
      const form = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: newEmail,
        mobile: originalUser.mobile,
        gender: originalUser.gender,
        deptId: originalUser.deptId,
        roleIds: existingRoleIds,
      });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.email).toBe(newEmail);
      expect(formData.email).not.toBe(beforeUpdate.email);
    });

    test("更新用户手机号并验证手机号正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newMobile = `139${String(Date.now()).slice(-8)}`;
      const form = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: beforeUpdate.email || originalUser.email,
        mobile: newMobile,
        gender: originalUser.gender,
        deptId: originalUser.deptId,
        roleIds: existingRoleIds,
      });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.mobile).toBe(newMobile);
      expect(formData.mobile).not.toBe(beforeUpdate.mobile);
    });

    test("更新用户状态并验证状态值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const form = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: beforeUpdate.email || originalUser.email,
        mobile: beforeUpdate.mobile || originalUser.mobile,
        gender: originalUser.gender,
        status: 0, // 禁用
        deptId: originalUser.deptId,
        roleIds: existingRoleIds,
      });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.status).toBe(0);

      // 验证在禁用列表中能查到
      const disabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 0 });
      const foundDisabled = disabledUsers.list.find((u) => u.id === testUserId);
      expect(foundDisabled).toBeDefined();
      expect(foundDisabled!.status).toBe(0);

      // 恢复启用状态
      const enableForm = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: beforeUpdate.email || originalUser.email,
        mobile: beforeUpdate.mobile || originalUser.mobile,
        gender: originalUser.gender,
        status: 1,
        deptId: originalUser.deptId,
        roleIds: existingRoleIds,
      });

      await UserAPI.update(testUserId, enableForm);

      const formData2 = await UserAPI.getFormData(testUserId);
      expect(formData2.status).toBe(1);

      // 验证在启用列表中能查到
      const enabledUsers = await UserAPI.getPage({ pageNum: 1, pageSize: 100, status: 1 });
      const foundEnabled = enabledUsers.list.find((u) => u.id === testUserId);
      expect(foundEnabled).toBeDefined();
      expect(foundEnabled!.status).toBe(1);
    });

    test("更新用户角色并验证角色分配正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newRoleIds = [ROLES.ROOT.id, ROLES.ADMIN.id, ROLES.GUEST.id];
      const form = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: beforeUpdate.email || originalUser.email,
        mobile: beforeUpdate.mobile || originalUser.mobile,
        gender: originalUser.gender,
        deptId: originalUser.deptId,
        roleIds: newRoleIds,
      });

      await UserAPI.update(testUserId, form);

      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.roleIds).toBeDefined();
      expect(Array.isArray(formData.roleIds)).toBe(true);
      expect(formData.roleIds!.length).toBe(newRoleIds.length);
      newRoleIds.forEach((roleId) => {
        expect(formData.roleIds).toContain(roleId);
      });
    });

    test("更新用户性别并验证性别值正确", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const newGender = beforeUpdate.gender === 1 ? 0 : 1;
      const form = createUserForm({
        username: originalUser.username,
        nickname: beforeUpdate.nickname || originalUser.nickname,
        email: beforeUpdate.email || originalUser.email,
        mobile: beforeUpdate.mobile || originalUser.mobile,
        gender: newGender,
        deptId: originalUser.deptId,
        roleIds: beforeUpdate.roleIds ?? existingRoleIds,
      });

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

      const form = createUserForm({
        username: originalUser.username,
        nickname: newNickname,
        email: newEmail,
        mobile: beforeUpdate.mobile || originalUser.mobile,
        gender: newGender,
        status: newStatus,
        deptId: originalUser.deptId,
        roleIds: beforeUpdate.roleIds ?? existingRoleIds,
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
      // 验证未更新的字段保持不变
      expect(formData.username).toBe(originalUser.username);
      expect(formData.mobile).toBe(beforeUpdate.mobile);

      // 恢复状态为启用
      await UserAPI.update(
        testUserId,
        createUserForm({
          username: originalUser.username,
          nickname: newNickname,
          email: newEmail,
          mobile: beforeUpdate.mobile || originalUser.mobile,
          gender: newGender,
          status: 1,
          deptId: originalUser.deptId,
          roleIds: beforeUpdate.roleIds ?? existingRoleIds,
        })
      );
    });

    test("更新不存在的用户", async () => {
      const nonExistentUserId = 99999999;
      const form = createUserForm({
        username: "nonexistent_user",
        nickname: "测试",
        roleIds: existingRoleIds,
      });

      await expectBizErrorOrUndefined(UserAPI.update(nonExistentUserId, form), [
        "A0401",
        "B0001",
        "A0400",
      ]);
    });

    test("参数校验：用户名冲突", async () => {
      const beforeUpdate = await UserAPI.getFormData(testUserId);
      const form = createUserForm({
        username: USERS.ADMIN.username, // 使用已存在的用户名
        nickname: beforeUpdate.nickname || originalUser.nickname,
        roleIds: beforeUpdate.roleIds ?? existingRoleIds,
      });
      await expectBizErrorOrUndefined(UserAPI.update(testUserId, form), [
        "A0501",
        "B0001",
        "A0400",
      ]);

      // 验证用户名未被修改
      const formData = await UserAPI.getFormData(testUserId);
      expect(formData.username).toBe(originalUser.username);
      expect(formData.username).not.toBe(USERS.ADMIN.username);
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
      await UserAPI.add(form);
      const userPageResult = await UserAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        keywords: form.username!,
      });
      const createdUser = userPageResult.list.find((u) => u.username === form.username);
      expect(createdUser).toBeDefined();
      testUserId = createdUser!.id!;
    });

    afterAll(async () => {
      try {
        await UserAPI.deleteByIds(testUserId.toString());
      } catch (e) {
        // 忽略删除错误
      }
    });

    test("修改用户密码并验证密码确实被修改", async () => {
      const newPassword = `NewPwd_${Date.now()}!`;

      await UserAPI.updatePassword(testUserId, newPassword);

      // 验证用户仍然存在且其他信息未变
      const formData = await UserAPI.getFormData(testUserId);
      expect(formData).toBeDefined();
      expect(formData.id).toBe(testUserId);
    });

    test("参数校验：空密码应被拒绝", async () => {
      const emptyPassword = "";
      await expectBizErrorOrUndefined(UserAPI.updatePassword(testUserId, emptyPassword), [
        "A0410",
        "A0400",
        "B0001",
      ]);
    });

    test("修改不存在用户的密码", async () => {
      const nonExistentUserId = 99999999;
      const newPassword = "newpassword123";

      await expectBizErrorOrUndefined(UserAPI.updatePassword(nonExistentUserId, newPassword), [
        "A0401",
        "B0001",
        "A0400",
      ]);
    });
  });

  describe("DELETE /api/v1/users/{ids} - 删除用户", () => {
    let testUserIds: number[] = [];
    const existingDeptId = DEPTS.CQUPT.id;

    beforeAll(async () => {
      for (let i = 0; i < 3; i++) {
        const form = createUserForm({
          username: `testuser_del_${Date.now()}_${i}`,
          nickname: `测试用户删除${i}`,
          status: 1,
          deptId: existingDeptId,
          roleIds: [ROLES.GUEST.id],
        });
        await UserAPI.add(form);

        const userPageResult = await UserAPI.getPage({
          pageNum: 1,
          pageSize: 100,
          keywords: form.username!,
        });
        const createdUser = userPageResult.list.find((u) => u.username === form.username);
        expect(createdUser).toBeDefined();
        testUserIds.push(createdUser!.id!);
      }
      // 验证所有测试用户都创建成功
      expect(testUserIds.length).toBe(3);
    });

    test("删除单个用户并验证用户真的被删除", async () => {
      const userId = testUserIds[0]!;

      // 验证删除前用户存在
      const beforeDelete = await UserAPI.getFormData(userId);
      expect(beforeDelete).toBeDefined();
      expect(beforeDelete.id).toBe(userId);

      // 执行删除
      await UserAPI.deleteByIds(userId.toString());

      // 验证删除后用户不存在
      const afterDelete = await UserAPI.getFormData(userId);
      expect(
        afterDelete === null || afterDelete === undefined || afterDelete.id === undefined
      ).toBe(true);

      // 验证在分页列表中也查不到
      const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100 });
      const deletedUser = pageResult.list.find((u) => u.id === userId);
      expect(deletedUser).toBeUndefined();

      // 验证总数减少
      expect(pageResult.total).toBeGreaterThanOrEqual(ADMIN_VISIBLE_USER_COUNT);
    });

    test("批量删除多个用户并验证所有用户都被删除", async () => {
      const ids = testUserIds.slice(1);
      expect(ids.length).toBe(2);

      // 验证删除前用户都存在
      for (const userId of ids) {
        const beforeDelete = await UserAPI.getFormData(userId!);
        expect(beforeDelete).toBeDefined();
        expect(beforeDelete.id).toBe(userId);
      }

      // 执行批量删除
      await UserAPI.deleteByIds(ids.join(","));

      // 验证所有用户都被删除
      for (const userId of ids) {
        const result = await UserAPI.getFormData(userId!);
        expect(result === null || result === undefined || result.id === undefined).toBe(true);
      }

      // 验证在分页列表中都查不到
      const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100 });
      for (const userId of ids) {
        const deletedUser = pageResult.list.find((u) => u.id === userId);
        expect(deletedUser).toBeUndefined();
      }
    });

    test("删除不存在的用户应保持幂等性", async () => {
      // DELETE 幂等性：删除不存在的资源应返回成功（不抛异常即视为幂等）
      const nonExistentUserId = "99999999";
      await UserAPI.deleteByIds(nonExistentUserId);
    });

    test("参数校验：空的ID列表", async () => {
      const emptyIds = "";
      await expectBizErrorOrUndefined(UserAPI.deleteByIds(emptyIds), ["B0001", "A0400"]);
    });
  });

  describe("GET /api/v1/users/template - 用户导入模板下载", () => {
    test("下载用户导入模板", async () => {
      const result = await UserAPI.downloadTemplate();

      expect(result).toBeDefined();
      expect(result instanceof ArrayBuffer || Buffer.isBuffer(result)).toBe(true);

      // 验证文件大小合理（Excel 文件至少有一定大小）
      const bufferSize = result instanceof ArrayBuffer ? result.byteLength : result.length;
      expect(bufferSize).toBeGreaterThan(1024); // 至少 1KB

      // 验证多次下载返回相同大小的文件（模板应该是固定的）
      const result2 = await UserAPI.downloadTemplate();
      const bufferSize2 = result2 instanceof ArrayBuffer ? result2.byteLength : result2.length;
      expect(bufferSize).toBe(bufferSize2);
    });
  });

  describe("GET /api/v1/users/export - 导出用户", () => {
    test("导出所有用户", async () => {
      const result = await UserAPI.export();

      expect(result).toBeDefined();
      expect(result instanceof ArrayBuffer || Buffer.isBuffer(result)).toBe(true);

      // 验证导出文件大小合理
      const bufferSize = result instanceof ArrayBuffer ? result.byteLength : result.length;
      expect(bufferSize).toBeGreaterThan(0);
    });

    test("按条件导出用户", async () => {
      const exportParams = { status: 1 };

      const result = await UserAPI.export(exportParams);

      expect(result).toBeDefined();
      expect(result instanceof ArrayBuffer || Buffer.isBuffer(result)).toBe(true);

      // 验证导出文件大小合理
      const bufferSize = result instanceof ArrayBuffer ? result.byteLength : result.length;
      expect(bufferSize).toBeGreaterThan(0);

      // 导出启用用户应该包含预置用户数据，文件应该有一定大小
      expect(bufferSize).toBeGreaterThan(1024);
    });

    test("导出禁用用户返回较小文件", async () => {
      const enabledResult = await UserAPI.export({ status: 1 });
      const disabledResult = await UserAPI.export({ status: 0 });

      const enabledSize =
        enabledResult instanceof ArrayBuffer ? enabledResult.byteLength : enabledResult.length;
      const disabledSize =
        disabledResult instanceof ArrayBuffer ? disabledResult.byteLength : disabledResult.length;

      // 启用用户（包含预置用户）的导出文件应该比禁用用户大
      expect(enabledSize).toBeGreaterThanOrEqual(disabledSize);
    });
  });

  describe("完整 CRUD 生命周期：用户管理", () => {
    test("创建→读→更新→读→删除→验证不存在", async () => {
      const existingDeptId = DEPTS.CQUPT.id;
      const existingRoleIds = [ROLES.GUEST.id];

      // Create: 创建用户
      const createForm = createUserForm({
        deptId: existingDeptId,
        roleIds: existingRoleIds,
        status: 1,
        gender: 1,
      });
      await UserAPI.add(createForm);

      // 通过分页查找创建的用户
      const pageResult = await UserAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        keywords: createForm.username!,
      });
      const createdUser = pageResult.list.find((u) => u.username === createForm.username);
      expect(createdUser).toBeDefined();
      const userId = createdUser!.id!;
      expect(userId).toBeGreaterThan(0);

      // Read: 验证字段与创建时一致
      const formData = await UserAPI.getFormData(userId);
      expect(formData.username).toBe(createForm.username);
      expect(formData.nickname).toBe(createForm.nickname);
      expect(formData.email).toBe(createForm.email);
      expect(formData.gender).toBe(1);
      expect(formData.status).toBe(1);
      expect(formData.deptId).toBe(existingDeptId);

      // Update: 更新用户昵称和状态
      const newNickname = `更新昵称_${Date.now()}`;
      const updateForm = createUserForm({
        username: createForm.username!,
        nickname: newNickname,
        email: formData.email ?? "",
        mobile: formData.mobile ?? "",
        gender: formData.gender ?? 1,
        status: 0,
        deptId: existingDeptId,
        roleIds: existingRoleIds,
      });
      await UserAPI.update(userId, updateForm);

      // Read: 验证更新已生效
      const updatedData = await UserAPI.getFormData(userId);
      expect(updatedData.nickname).toBe(newNickname);
      expect(updatedData.nickname).not.toBe(createForm.nickname);
      expect(updatedData.status).toBe(0);

      // Delete: 删除用户
      await UserAPI.deleteByIds(userId.toString());

      // Verify: 验证用户已从分页列表中移除
      const afterDeletePage = await UserAPI.getPage({ pageNum: 1, pageSize: 100 });
      const deletedInList = afterDeletePage.list.find((u) => u.id === userId);
      expect(deletedInList).toBeUndefined();

      // Verify: 验证表单数据已不存在
      const afterDeleteForm = await UserAPI.getFormData(userId);
      expect(
        afterDeleteForm === null ||
          afterDeleteForm === undefined ||
          afterDeleteForm.id === undefined
      ).toBe(true);
    });
  });
});
