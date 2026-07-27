// 测试数据常量——存在于测试数据库中，一般不会变化
export const USERS = {
  ROOT: {
    id: 1,
    username: "root",
    nickname: "root",
    gender: 0,
    status: 1,
    deptId: null as number | null,
    mobile: "18838027307",
    email: "1066365803@qq.com",
    roleIds: [1] as number[], // ROOT 角色
  },
  ADMIN: {
    id: 2,
    username: "admin",
    nickname: "admin",
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
    nickname: "测试用户",
    gender: 1,
    status: 1,
    deptId: 3,
    mobile: "19122145917",
    email: "w1066365803@icloud.com",
    roleIds: [3] as number[], // GUEST 角色
  },
  DEPT_ADMIN: {
    id: 4,
    username: "dept_admin",
    nickname: "部门管理员",
    gender: 1,
    status: 1,
    deptId: 2,
    mobile: "13800000004",
    email: "dept_admin@dehaze.com",
    roleIds: [4] as number[], // DEPT_ADMIN 角色
  },
  USER: {
    id: 5,
    username: "user",
    nickname: "普通用户",
    gender: 1,
    status: 1,
    deptId: 2,
    mobile: "13800000005",
    email: "user@dehaze.com",
    roleIds: [5] as number[], // USER 角色
    member: { levelCode: "level_0" as const, growthValue: 100 },
  },
  VIP1: {
    id: 6,
    username: "vip1",
    nickname: "VIP1用户",
    gender: 1,
    status: 1,
    deptId: 2,
    mobile: "13800000006",
    email: "vip1@dehaze.com",
    roleIds: [5] as number[], // USER 角色
    member: { levelCode: "level_1" as const, growthValue: 1500 },
  },
  VIP2: {
    id: 7,
    username: "vip2",
    nickname: "VIP2用户",
    gender: 2,
    status: 1,
    deptId: 2,
    mobile: "13800000007",
    email: "vip2@dehaze.com",
    roleIds: [5] as number[], // USER 角色
    member: { levelCode: "level_2" as const, growthValue: 8000 },
  },
  SVIP: {
    id: 8,
    username: "svip",
    nickname: "SVIP用户",
    gender: 1,
    status: 1,
    deptId: 2,
    mobile: "13800000008",
    email: "svip@dehaze.com",
    roleIds: [5] as number[], // USER 角色
    member: { levelCode: "level_3" as const, growthValue: 25000 },
  },
};

export const ROLES = {
  ROOT: { id: 1, code: "ROOT", name: "超级管理员" },
  ADMIN: { id: 2, code: "ADMIN", name: "系统管理员" },
  GUEST: { id: 3, code: "GUEST", name: "访问游客" },
  DEPT_ADMIN: { id: 4, code: "DEPT_ADMIN", name: "部门管理员" },
  USER: { id: 5, code: "USER", name: "普通用户" },
} as const;

export const DEPTS = {
  CQUPT: { id: 1, name: "重庆邮电大学", parentId: 0 },
  SOFTWARE: { id: 2, name: "软件工程学院", parentId: 1 },
  COMPUTER: { id: 3, name: "计算机学院", parentId: 1 },
} as const;

// 预置用户数量（不含测试创建的用户）
export const PRESET_USER_COUNT = 8;

// admin 用户可见的预置用户数量
// ADMIN 角色 data_scope=0 (ALL)，不受数据权限过滤，可见所有预置用户
export const ADMIN_VISIBLE_USER_COUNT = 8;

// 按会员等级索引用户，便于按等级筛选测试
export const USERS_BY_LEVEL = {
  level_0: USERS.USER,
  level_1: USERS.VIP1,
  level_2: USERS.VIP2,
  level_3: USERS.SVIP,
} as const;
