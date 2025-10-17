/**
 * 分页结果基类
 */
export class PageResult<T> {
  /**
   * 数据列表
   */
  list: Array<T> = [];
  
  /**
   * 总记录数
   */
  total: number = 0;
}