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
  msgContains?: string | string[]
): Promise<void> {
  const codes = Array.isArray(expectedCode) ? expectedCode : [expectedCode];

  await expect(promise).rejects.toSatisfy((error: any) => {
    const bizError = getBizError(error);
    // 允许 axios 错误码（以 ERR_ 开头）或在期望列表中
    const isAxiosError = bizError.code?.startsWith("ERR_") ?? false;
    const codeMatch = codes.includes(bizError.code || "") || isAxiosError;

    // 如果是 axios 错误，跳过消息匹配（axios 错误消息是通用的）
    let msgMatch = true;
    if (msgContains && !isAxiosError) {
      const msgs = Array.isArray(msgContains) ? msgContains : [msgContains];
      msgMatch = msgs.some((msg) => bizError.msg?.includes(msg) ?? false);
    }

    return codeMatch && msgMatch;
  });
}

/**
 * 断言 Promise 会抛出业务错误（允许错误码为 undefined 或 axios 错误码）
 * 用于某些参数校验场景，错误码可能是 A0400、B0001、ERR_BAD_REQUEST 或 undefined
 * @param promise 待测试的 Promise
 * @param expectedCodes 期望的错误码列表
 * @param msgContains 错误消息应包含的文本（可选，支持单个字符串或字符串数组）
 */
export async function expectBizErrorOrUndefined(
  promise: Promise<any>,
  expectedCodes: string[],
  msgContains?: string | string[]
): Promise<void> {
  await expect(promise).rejects.toSatisfy((error: any) => {
    const bizError = getBizError(error);
    // 允许 code 为 undefined、在期望列表中、或是 axios 错误码（以 ERR_ 开头）
    const isAxiosError = bizError.code?.startsWith("ERR_") ?? false;
    const codeMatch = !bizError.code || expectedCodes.includes(bizError.code) || isAxiosError;

    // 如果是 axios 错误，跳过消息匹配（axios 错误消息是通用的）
    let msgMatch = true;
    if (msgContains && !isAxiosError) {
      const msgs = Array.isArray(msgContains) ? msgContains : [msgContains];
      msgMatch = msgs.some((msg) => bizError.msg?.includes(msg) ?? false);
    }

    return codeMatch && msgMatch;
  });
}
