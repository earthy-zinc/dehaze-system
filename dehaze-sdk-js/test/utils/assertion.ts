import { expect } from "vitest";

/**
 * 断言 Promise 会抛出业务错误
 * 提供统一的业务错误断言方法，替代手动 throw new Error("应该抛出异常") 的反模式
 * 匹配规则：
 * - 携带业务错误码时，错误码需命中 expectedCode；
 * - 为传输层错误（无业务错误码，或 code 以 ERR_ 开头，如 HTTP 400 / 网络错误）时直接放行——
 *   不同后端（Java/Go/Python）对同一异常场景返回的错误码与错误体格式不一致，测试不应断言传输层细节。
 *
 * @param promise 待测试的 Promise
 * @param expectedCode 期望的业务错误码（单个或多个，对应不同后端可能的返回）
 * @param msgContains 错误消息应包含的文本（可选，支持单个字符串或字符串数组；仅对非传输层错误生效）
 */
export async function expectBizError(
  promise: Promise<any>,
  expectedCode: string | string[],
  msgContains?: string | string[]
): Promise<void> {
  const codes = Array.isArray(expectedCode) ? expectedCode : [expectedCode];

  await expect(promise).rejects.toSatisfy((error: any) => {
    // 提取业务错误信息：优先取 axios response.data，其次取 error.code，最后回退到 error 本身
    let bizError: { code?: string; msg?: string };
    if (error?.response?.data) {
      bizError = error.response.data;
    } else if (error && typeof error.code === "string") {
      bizError = { code: error.code, msg: error.message };
    } else {
      bizError = error || {};
    }

    const code = bizError.code;
    const isAxiosError = code?.startsWith("ERR_") ?? false;
    // 业务码命中期望集合，或为传输层错误（无业务码 / ERR_ 前缀）即视为码匹配
    const codeMatch = !code || isAxiosError || codes.includes(code);

    let msgMatch = true;
    // 仅在非 axios 传输错误时校验消息
    if (msgContains && !isAxiosError) {
      const msgs = Array.isArray(msgContains) ? msgContains : [msgContains];
      msgMatch = msgs.some((msg) => bizError.msg?.includes(msg) ?? false);
    }

    return codeMatch && msgMatch;
  });
}
