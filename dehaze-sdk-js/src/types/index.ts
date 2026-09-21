/**
 * 响应数据
 */
export interface ResponseData<T = any> {
  code: string;
  data: T;
  msg: string;
  traceId: string;
}

/**
 * 请求数据，用于失败排查时还原「发了什么请求」
 */
export interface RequestData {
  method: string;
  url: string;
  params?: unknown;
  body?: unknown;
}

/**
 * 分页查询参数
 */
export interface PageQuery {
  pageNum?: number;
  pageSize?: number;
}

/**
 * 分页响应对象
 */
export interface PageResult<T> {
  /** 数据列表 */
  list: T;
  /** 总数 */
  total: number;
}

/**
 * 游标分页响应对象（按 id 倒序游标翻页，用于聊天历史向上增量加载）
 */
export interface CursorResult<T> {
  /** 数据列表 */
  list: T;
  /** 总数（符合游标条件的总条数） */
  total: number;
  /** 是否还有更早数据（存在 id 小于本页最小 id 的消息） */
  hasMore: boolean;
}

/**
 * 通用启用状态：1-启用，0-禁用
 */
export type EnabledStatus = 0 | 1;

/**
 * 组件数据源
 */
export interface OptionType {
  /** 值 */
  value: string | number;
  /** 文本 */
  label: string;
  /** 子列表  */
  children?: OptionType[];
}
