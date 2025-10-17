/**
 * 登录用户信息
 */
export class UserInfo {
  userId?: number = undefined;
  username?: string = undefined;
  nickname?: string = undefined;
  avatar?: string = undefined;
  roles: Array<string> = [];
  perms: Array<string> = [];
}

/**
 * 用户查询对象类型
 */
export class UserQuery {
  keywords?: string = undefined;
  status?: number = undefined;
  deptId?: number = undefined;
  startTime?: string = undefined;
  endTime?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
}

/**
 * 用户分页对象
 */
export class UserPageVO {
  /**
   * 用户头像地址
   */
  avatar?: string = undefined;
  /**
   * 创建时间
   */
  createTime?: string = undefined;
  /**
   * 部门名称
   */
  deptName?: string = undefined;
  /**
   * 用户邮箱
   */
  email?: string = undefined;
  /**
   * 性别
   */
  genderLabel?: string = undefined;
  /**
   * 用户ID
   */
  id?: number = undefined;
  /**
   * 手机号
   */
  mobile?: string = undefined;
  /**
   * 用户昵称
   */
  nickname?: string = undefined;
  /**
   * 角色名称，多个使用英文逗号(,)分割
   */
  roleNames?: string = undefined;
  /**
   * 用户状态(1:启用;0:禁用)
   */
  status?: number = undefined;
  /**
   * 用户名
   */
  username?: string = undefined;
}

/**
 * 用户表单类型
 */
export class UserForm {
  /**
   * 用户头像
   */
  avatar?: string = undefined;
  /**
   * 部门ID
   */
  deptId?: number = undefined;
  /**
   * 邮箱
   */
  email?: string = undefined;
  /**
   * 性别
   */
  gender?: number = undefined;
  /**
   * 用户ID
   */
  id?: number = undefined;
  mobile?: string = undefined;
  /**
   * 昵称
   */
  nickname?: string = undefined;
  /**
   * 角色ID集合
   */
  roleIds?: Array<number> = undefined;
  /**
   * 用户状态(1:正常;0:禁用)
   */
  status?: number = undefined;
  /**
   * 用户名
   */
  username?: string = undefined;
}