/**
 * 业务逻辑相关的测试辅助工具
 * 提供业务错误处理等功能
 */

/**
 * 从错误对象中提取业务错误信息
 * 响应拦截器会将业务错误直接reject为{code, msg}对象
 * 但axios错误会包含response.data
 */
export function getBizError(error: any): { code?: string; msg?: string } {
  // 如果error本身有code属性且是字符串，说明是响应拦截器处理过的业务错误
  if (error && typeof error.code === "string") {
    return error;
  }
  // 尝试从axios错误中获取
  if (error?.response?.data) {
    return error.response.data;
  }
  // 如果是axios错误，可能有code属性（如ERR_BAD_REQUEST）
  if (error?.code) {
    return { code: error.code, msg: error.message };
  }
  return error || {};
}
