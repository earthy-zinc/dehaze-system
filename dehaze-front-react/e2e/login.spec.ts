import { expect, test } from "@playwright/test";

import { SEED_PASSWORD, capturePageCaptcha } from "./utils/auth";

// 本文件验证登录流程本身，需要以未登录态启动
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("用户登录流程", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/login");
  });

  test("登录页正常渲染", async ({ page }) => {
    await expect(page.locator(".login-form")).toBeVisible();
    await expect(page.locator('input[placeholder="用户名"]')).toBeVisible();
    await expect(page.locator('input[placeholder="密码"]')).toBeVisible();
  });

  test("空表单提交提示必填错误", async ({ page }) => {
    await page.locator('.login-form button[type="submit"]').click();

    await expect(
      page.locator(".ant-form-item-explain-error").first()
    ).toBeVisible();
  });

  test("无效凭据提示错误", async ({ page }) => {
    const captchaCode = await capturePageCaptcha(page);

    await page.locator('input[placeholder="用户名"]').fill("invalid_user");
    await page.locator('input[placeholder="密码"]').fill("wrong_password");
    await page.locator('input[placeholder="验证码"]').fill(captchaCode);
    await page.locator('.login-form button[type="submit"]').click();

    await expect(page.locator(".ant-message-notice").first()).toBeVisible({
      timeout: 15000,
    });
  });

  test("有效凭据登录成功并离开登录页", async ({ page }) => {
    const captchaCode = await capturePageCaptcha(page);

    await page.locator('input[placeholder="用户名"]').fill("admin");
    await page.locator('input[placeholder="密码"]').fill(SEED_PASSWORD);
    await page.locator('input[placeholder="验证码"]').fill(captchaCode);
    await page.locator('.login-form button[type="submit"]').click();

    await page.waitForURL((url) => new URL(url).pathname !== "/login", {
      timeout: 30000,
    });
    await expect(page.locator(".login-container")).toBeHidden();
  });
});
