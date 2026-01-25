// 测试数据常量——存在于测试数据库中，一般不会变化
export const USERS = {
  ROOT: {
    id: 1,
    username: "root",
    nickname: "有来技术",
    gender: 0,
    status: 1,
    deptId: null as number | null,
    mobile: "17621590365",
    email: "youlaitech@163.com",
    roleIds: [1] as number[], // ROOT 角色
  },
  ADMIN: {
    id: 2,
    username: "admin",
    nickname: "武沛鑫",
    gender: 1,
    status: 1,
    deptId: 1,
    email: "w1066365803@163.com",
    mobile: "18537958917",
    roleIds: [2] as number[], // ADMIN 角色
  },
  TEST: {
    id: 3,
    username: "test",
    nickname: "测试小用户",
    gender: 1,
    status: 1,
    deptId: 3,
    mobile: "17621210366",
    email: "youlaitech@163.com",
    roleIds: [3] as number[], // GUEST 角色
  },
};

export const ROLES = {
  ROOT: { id: 1, code: "ROOT", name: "超级管理员" },
  ADMIN: { id: 2, code: "ADMIN", name: "系统管理员" },
  GUEST: { id: 3, code: "GUEST", name: "访问游客" },
} as const;

export const DEPTS = {
  CQUPT: { id: 1, name: "重庆邮电大学", parentId: 0 },
  SOFTWARE: { id: 2, name: "软件工程学院", parentId: 1 },
  COMPUTER: { id: 3, name: "计算机学院", parentId: 1 },
} as const;

// 预置用户数量（不含测试创建的用户）
export const PRESET_USER_COUNT = 3;

// admin 用户可见的预置用户数量（受数据权限限制，root 用户 deptId=null 不可见）
export const ADMIN_VISIBLE_USER_COUNT = 2;
