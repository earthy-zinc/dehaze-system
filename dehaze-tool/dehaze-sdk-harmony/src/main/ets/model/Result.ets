/**
 * 通用响应结果包装类
 *
 * 与后端 common.Response 结构对齐（API 规范 04-API规范.md §3.1）：
 * - code: string，成功为 "00000"
 * - msg: string，状态描述
 * - data: 业务数据
 */
export class Result<T> {
  /** 业务状态码，成功为 "00000" */
  code: string = '00000';
  /** 状态描述信息 */
  msg: string = '';
  /** 业务数据 */
  data: T = undefined;
  /** 请求追踪 ID */
  traceId?: string = undefined;
  /** 响应时间戳（毫秒） */
  timestamp?: number = undefined;
  /** 错误详情列表（参数校验失败时） */
  errors?: Array<ErrorItem> = undefined;

  /** 是否成功 */
  isSuccess(): boolean {
    return this.code === '00000';
  }
}

/** 参数校验错误项 */
export class ErrorItem {
  field: string = '';
  message: string = '';
  code: string = '';
}

/** 业务状态码枚举（与后端 common.ResultCode 对齐） */
export class ResultCode {
  /** 成功 */
  static readonly SUCCESS: string = '00000';

  /** Token 无效或已过期 */
  static readonly TOKEN_INVALID: string = 'A0230';
  /** Token 已被禁止访问 */
  static readonly TOKEN_ACCESS_FORBIDDEN: string = 'A0231';
  /** 访问未授权 */
  static readonly ACCESS_UNAUTHORIZED: string = 'A0301';

  /** 用户名或密码错误 */
  static readonly USERNAME_OR_PASSWORD_ERROR: string = 'A0210';
  /** 验证码已过期 */
  static readonly VERIFY_CODE_TIMEOUT: string = 'A0213';
  /** 验证码错误 */
  static readonly VERIFY_CODE_ERROR: string = 'A0214';

  /** 用户请求参数错误 */
  static readonly PARAM_ERROR: string = 'A0400';
  /** 系统执行出错 */
  static readonly SYSTEM_EXECUTION_ERROR: string = 'B0001';
}

/** Token 失效相关错误码，触发自动刷新或跳转登录 */
export const TOKEN_INVALID_CODES: string[] = [
  ResultCode.TOKEN_INVALID,
  ResultCode.TOKEN_ACCESS_FORBIDDEN,
  ResultCode.ACCESS_UNAUTHORIZED,
];
