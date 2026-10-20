import { test as setup } from "@playwright/test";

import { loginByApi } from "./utils/auth";

/** 登录态落盘供后续用例复用：一次登录替代每个用例各登录一遍 */
export const AUTH_FILE = "test_result/.auth/user.json";

setup("以种子账号建立会话", async ({ context }) => {
  await loginByApi(context);
  await context.storageState({ path: AUTH_FILE });
});
