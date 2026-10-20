import { expect, test } from "@playwright/test";

import { SEED_PASSWORD, capturePageCaptcha, loginByApi } from "./utils/auth";

// 本文件验证登录流程本身，需要以未登录态启动
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("用户登录流程", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/login");
  });

  test("未登录冷启动不把 /auth/me 401 记为 ERROR 日志", async ({ page }) => {
    const logs: string[] = [];
    page.on("console", (message) =>
      logs.push(`[${message.type()}] ${message.text()}`)
    );

    await page.goto("/#/login");
    // 冷启动探测确实发生（否则下面的断言是空跑），且只能记为 INFO
    await expect
      .poll(() => logs.find((text) => text.includes("API_UNAUTHORIZED")), {
        message: "未产生冷启动 /auth/me 401 的 INFO 日志",
      })
      .toBeTruthy();
    expect(logs.filter((text) => text.includes("API_ERROR"))).toEqual([]);
  });

  test("登录页正常渲染", async ({ page }) => {
    await expect(page.locator(".login-form")).toBeVisible();
    await expect(page.locator('input[name="username"]')).toBeVisible();
    await expect(page.locator('input[name="password"]')).toBeVisible();
  });

  test("空表单提交提示必填错误", async ({ page }) => {
    await page.locator(".login-form button").click();

    await expect(page.locator(".el-form-item__error").first()).toBeVisible();
  });

  test("无效凭据提示错误", async ({ page }) => {
    const captchaCode = await capturePageCaptcha(page);

    await page.locator('input[name="username"]').fill("invalid_user");
    await page.locator('input[name="password"]').fill("wrong_password");
    await page.locator(".login-form input").nth(2).fill(captchaCode);
    await page.locator(".login-form button").click();

    await expect(page.locator(".el-message--error").first()).toBeVisible({
      timeout: 15000,
    });
  });

  test("有效凭据登录成功并跳转首页", async ({ page }) => {
    const captchaCode = await capturePageCaptcha(page);

    await page.locator('input[name="username"]').fill("admin");
    await page.locator('input[name="password"]').fill(SEED_PASSWORD);
    await page.locator(".login-form input").nth(2).fill(captchaCode);
    await page.locator(".login-form button").click();

    await page.waitForURL(/#\/home/, { timeout: 30000 });
    await expect(page.locator(".main-container")).toBeVisible();
  });

  test("登录后可登出并回到登录页", async ({ page }) => {
    await loginByApi(page.context());
    await page.goto("/#/home");
    await page.locator(".main-container").waitFor();

    // 用户名下拉（唯一菜单项为退出登录）→ 注销确认
    await page.locator(".el-dropdown").filter({ hasText: "admin" }).click();
    await page.locator(".el-dropdown-menu__item").click();
    await page.locator(".el-message-box .el-button--primary").click();

    await page.waitForURL(/#\/login/, { timeout: 30000 });
    await expect(page.locator(".login-form")).toBeVisible();
  });
});
