/**
 * 部门查询参数
 */
export class DeptQuery {
  keywords?: string = undefined;
  status?: number = undefined;
}

/**
 * 部门类型
 */
export class DeptVO {
  /**
   * 子部门
   */
  children?: Array<DeptVO> = undefined;
  /**
   * 创建时间
   */
  createTime?: string = undefined;
  /**
   * 部门ID
   */
  id?: number = undefined;
  /**
   * 部门名称
   */
  name?: string = undefined;
  /**
   * 父部门ID
   */
  parentId?: number = undefined;
  /**
   * 排序
   */
  sort?: number = undefined;
  /**
   * 状态(1:启用；0:禁用)
   */
  status?: number = undefined;
  /**
   * 修改时间
   */
  updateTime?: string = undefined;
}

/**
 * 部门表单类型
 */
export class DeptForm {
  /**
   * 部门ID(新增不填)
   */
  id?: number = undefined;
  /**
   * 部门名称
   */
  name?: string = undefined;
  /**
   * 父部门ID
   */
  parentId: number = 0;
  /**
   * 排序
   */
  sort?: number = undefined;
  /**
   * 状态(1:启用；0：禁用)
   */
  status?: number = undefined;
}