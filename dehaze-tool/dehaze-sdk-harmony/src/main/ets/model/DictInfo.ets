/**
 * 字典类型查询参数
 */
export class DictTypeQuery {
  /**
   * 关键字(字典类型名称/编码)
   */
  keywords?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
}

/**
 * 字典类型分页对象
 */
export class DictTypePageVO {
  /**
   * 字典类型ID
   */
  id: number = 0;
  /**
   * 类型编码
   */
  code: string = '';
  /**
   * 类型名称
   */
  name: string = '';
  /**
   * 状态(1:启用;0:禁用)
   */
  status?: number = undefined;
  /**
   * 备注
   */
  remark?: string = undefined;
}

/**
 * 字典表单类型声明
 */
export class DictTypeForm {
  /**
   * 字典类型ID
   */
  id?: number = undefined;
  /**
   * 类型名称
   */
  name?: string = undefined;
  /**
   * 类型编码
   */
  code?: string = undefined;
  /**
   * 类型状态：1:启用;0:禁用
   */
  status: number = 1;
  /**
   * 备注
   */
  remark?: string = undefined;
}

/**
 * 字典查询参数
 */
export class DictQuery {
  /**
   * 字典项名称
   */
  name?: string = undefined;
  /**
   * 字典类型编码
   */
  typeCode?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
}

/**
 * 字典分页对象
 */
export class DictPageVO {
  /**
   * 字典ID
   */
  id?: number = undefined;
  /**
   * 字典名称
   */
  name?: string = undefined;
  /**
   * 状态(1:启用;0:禁用)
   */
  status?: number = undefined;
  /**
   * 字典值
   */
  value?: string = undefined;
}

/**
 * 字典表单
 */
export class DictForm {
  /**
   * 字典ID
   */
  id?: number = undefined;
  /**
   * 字典名称
   */
  name?: string = undefined;
  /**
   * 排序
   */
  sort?: number = undefined;
  /**
   * 状态(1:启用;0:禁用)
   */
  status?: number = undefined;
  /**
   * 类型编码
   */
  typeCode?: string = undefined;
  /**
   * 值
   */
  value?: string = undefined;

  /**
   * 备注
   */
  remark?: string = undefined;
}