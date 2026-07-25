/**
 * 测试错误断言辅助函数
 *
 * 提供统一的业务错误断言方法，替代手动 throw new Error("应该抛出异常") 的反模式
 */
import { expect } from "vitest";
import { getBizError } from "#/utils/biz";

/**
 * 断言 Promise 会抛出业务错误
 * @param promise 待测试的 Promise
 * @param expectedCode 期望的错误码（支持单个或多个）
 * @param msgContains 错误消息应包含的文本（可选，支持单个字符串或字符串数组）
 */
export async function expectBizError(
  promise: Promise<any>,
  expectedCode: string | string[],
  msgContains?: string | string[],
  allowUndefinedCode = false
): Promise<void> {
  const codes = Array.isArray(expectedCode) ? expectedCode : [expectedCode];

  await expect(promise).rejects.toSatisfy((error: any) => {
    const bizError = getBizError(error);
    const isAxiosError = bizError.code?.startsWith("ERR_") ?? false;
    const codeMatch = allowUndefinedCode
      ? !bizError.code || codes.includes(bizError.code) || isAxiosError
      : codes.includes(bizError.code || "") || isAxiosError;

    let msgMatch = true;
    if (msgContains && !isAxiosError) {
      const msgs = Array.isArray(msgContains) ? msgContains : [msgContains];
      msgMatch = msgs.some((msg) => bizError.msg?.includes(msg) ?? false);
    }

    return codeMatch && msgMatch;
  });
}
