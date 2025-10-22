/**
 * 角色查询参数
 */
export class RoleQuery {
  keywords?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
}

/**
 * 角色分页对象
 */
export class RolePageVO {
  /**
   * 角色编码
   */
  code?: string = undefined;

  /**
   * 角色ID
   */
  id?: number = undefined;
  /**
   * 角色名称
   */
  name?: string = undefined;
  /**
   * 排序
   */
  sort?: number = undefined;
  /**
   * 角色状态
   */
  status?: number = undefined;
  /**
   * 创建时间
   */
  createTime?: string = undefined;
  /**
   * 修改时间
   */
  updateTime?: string = undefined;
}

/**
 * 角色表单对象
 */
export class RoleForm {
  /**
   * 角色ID
   */
  id?: number = undefined;

  /**
   * 角色编码
   */
  code: string = '';
  /**
   * 数据权限
   */
  dataScope?: number = undefined;

  /**
   * 角色名称
   */
  name: string = '';
  /**
   * 排序
   */
  sort?: number = undefined;
  /**
   * 角色状态(1-正常；0-停用)
   */
  status?: number = undefined;
}