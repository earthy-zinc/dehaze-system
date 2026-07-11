/**
 * API 响应码枚举
 * 与 dehaze-sdk-js ResultEnum 保持一致
 */
export const ResultCode = {
  /** 成功 */
  SUCCESS: "00000",
  /** Token 无效或已过期 */
  TOKEN_INVALID: "A0230",
  /** Token 已被禁止访问 */
  TOKEN_ACCESS_FORBIDDEN: "A0231",
  /** 用户不存在 */
  USER_NOT_EXIST: "A0201",
  /** 用户名或密码错误 */
  USERNAME_OR_PASSWORD_ERROR: "A0210",
  /** 验证码已过期 */
  VERIFY_CODE_TIMEOUT: "A0213",
  /** 验证码错误 */
  VERIFY_CODE_ERROR: "A0214",
  /** 访问未授权 */
  ACCESS_UNAUTHORIZED: "A0301",
  /** 演示环境禁止操作 */
  FORBIDDEN_OPERATION: "A0302",
} as const;

export type ResultCodeValue = (typeof ResultCode)[keyof typeof ResultCode];

/** Token 相关需要重新登录的错误码 */
export const TOKEN_INVALID_CODES: ResultCodeValue[] = [
  ResultCode.TOKEN_INVALID,
  ResultCode.TOKEN_ACCESS_FORBIDDEN,
];
