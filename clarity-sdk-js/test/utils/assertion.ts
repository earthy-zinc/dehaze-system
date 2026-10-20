import { expect } from "vitest";

/**
 * 断言 Promise 会抛出业务错误
 * 提供统一的业务错误断言方法，替代手动 throw new Error("应该抛出异常") 的反模式
 * 匹配规则：
 * - 错误必须携带业务错误码且命中 expectedCode（严格匹配）；
 * - 无业务错误码或传输层错误（code 以 ERR_ 开头）均视为断言失败——
 *   传输层错误说明后端未按契约返回业务错误码信封，属于需要暴露的契约缺陷，而非放行通过。
 *
 * @param promise 待测试的 Promise
 * @param expectedCode 期望的业务错误码（单个或多个）
 * @param msgContains 错误消息应包含的文本（可选，支持单个字符串或字符串数组）
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

    const code = bizError.code!;
    const codeMatch = codes.includes(code);

    let msgMatch = true;
    if (msgContains) {
      const msgs = Array.isArray(msgContains) ? msgContains : [msgContains];
      msgMatch = msgs.some((msg) => bizError.msg?.includes(msg) ?? false);
    }

    return codeMatch && msgMatch;
  });
}
