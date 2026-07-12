/**
 * 分页结果（后端 vo.PageResult / API 规范 §3.3）
 */
export class PageResult<T> {
  /** 当前页数据列表 */
  list: T[] = [];
  /** 总记录数 */
  total: number = 0;
}
