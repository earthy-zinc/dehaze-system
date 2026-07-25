/**
 * 业务逻辑相关的测试辅助工具
 * 提供业务错误处理等功能
 */

/**
 * 从错误对象中提取业务错误信息
 *
 * 优先从 axios 错误的 response.data 中提取业务错误码和消息，
 * 这样无论后端返回 HTTP 200（业务错误码在 body 中）还是 HTTP 400（业务错误码也在 body 中），
 * 都能正确提取真实的业务错误码。
 */
export function getBizError(error: any): { code?: string; msg?: string } {
  // 优先从 axios 错误的 response.data 中提取业务错误
  // 适用于：HTTP 400/422 等场景（Python 后端），以及 SDK 拦截器构造的 mock error（HTTP 200 业务错误）
  if (error?.response?.data) {
    return error.response.data;
  }
  // 如果 error 本身有 code 属性且是字符串（如网络错误 ERR_NETWORK）
  if (error && typeof error.code === "string") {
    return { code: error.code, msg: error.message };
  }
  return error || {};
}
