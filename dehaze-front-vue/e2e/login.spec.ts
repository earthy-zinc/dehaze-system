import { expect, test } from "@playwright/test";

test.describe("用户登录流程", () => {
  test.beforeEach(async ({ page }) => {
    // 访问登录页面
    await page.goto("/login");
  });

  test("应该成功显示登录页面", async ({ page }) => {
    // 验证页面标题
    await expect(page).toHaveTitle(/去雾系统|Dehaze System/i);

    // 验证登录表单元素存在
    await expect(page.locator('input[type="text"]').first()).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    // 更新选择器以匹配实际的登录按钮
    await expect(
      page.locator('.login-form button[type="button"]').first()
    ).toBeVisible();
  });

  test("应该验证必填字段", async ({ page }) => {
    // 点击登录按钮而不填写任何内容
    await page.locator('.login-form button[type="button"]').first().click();

    // 验证错误消息显示
    await expect(page.locator(".el-form-item__error").first()).toBeVisible();
  });

  test("应该处理无效的用户名和密码", async ({ page }) => {
    // 填写无效的用户名和密码
    await page.locator('input[type="text"]').first().fill("invalid_user");
    await page.locator('input[type="password"]').fill("wrong_password");
    // 填写验证码
    await page.locator('input[type="text"]').nth(2).fill("12345678");

    // 点击登录按钮
    await page.locator('.login-form button[type="button"]').first().click();

    // 等待错误消息
    await page.waitForTimeout(1000);

    // 验证错误提示（根据实际实现调整选择器）
    const errorMessage = page.locator(".el-message--error, .el-notification");
    await expect(errorMessage).toBeVisible({ timeout: 15000 });
  });

  test("应该成功登录并跳转到首页", async ({ page }) => {
    // 填写有效的用户名和密码（根据页面提示）
    await page.locator('input[type="text"]').first().fill("admin");
    await page.locator('input[type="password"]').fill("Dehaze2026");
    // 填写验证码
    await page.locator('input[type="text"]').nth(2).fill("12345678");

    // 点击登录按钮
    await page.locator('.login-form button[type="button"]').first().click();

    // 等待导航到首页
    await page.waitForURL(/\/(home|dashboard|index)/, { timeout: 30000 });

    // 验证成功登录后的元素存在
    await expect(page.locator(".layout-header, .el-header")).toBeVisible({
      timeout: 15000,
    });

    // 验证用户信息显示
    await expect(
      page.locator('.user-info, [class*="user"], [class*="avatar"]')
    ).toBeVisible({ timeout: 15000 });
  });

  test("应该记住用户名（如果勾选记住我）", async ({ page, context }) => {
    // 填写用户名和密码
    await page.locator('input[type="text"]').first().fill("testuser");
    await page.locator('input[type="password"]').fill("password123");
    // 填写验证码
    await page.locator('input[type="text"]').nth(2).fill("12345678");

    // 勾选"记住我"复选框（如果存在）
    const rememberCheckbox = page.locator('input[type="checkbox"]').first();
    if (await rememberCheckbox.isVisible()) {
      await rememberCheckbox.check();
    }

    // 点击登录
    await page.locator('.login-form button[type="button"]').first().click();

    // 等待页面跳转
    await page.waitForTimeout(2000);

    // 检查 cookies 或 localStorage 中是否保存了用户名
    const cookies = await context.cookies();
    const hasUsernameCookie = cookies.some(
      (cookie) => cookie.name === "username" || cookie.name === "remember_user"
    );

    const localStorageUsername = await page.evaluate(() => {
      return localStorage.getItem("username") || localStorage.getItem("user");
    });

    // 至少有一种方式保存了用户信息
    expect(hasUsernameCookie || !!localStorageUsername).toBeTruthy();
  });

  test("应该支持通过键盘 Enter 键登录", async ({ page }) => {
    // 填写用户名和密码
    await page.locator('input[type="text"]').first().fill("admin");
    await page.locator('input[type="password"]').fill("Dehaze2026");
    // 填写验证码
    await page.locator('input[type="text"]').nth(2).fill("12345678");

    // 在密码框按 Enter 键
    await page.locator('input[type="password"]').press("Enter");

    // 等待导航
    await page.waitForURL(/\/(home|dashboard|index)/, { timeout: 30000 });

    // 验证登录成功
    await expect(page.locator(".layout-header, .el-header")).toBeVisible({
      timeout: 15000,
    });
  });

  test("应该支持登出功能", async ({ page }) => {
    // 先登录
    await page.locator('input[type="text"]').first().fill("admin");
    await page.locator('input[type="password"]').fill("Dehaze2026");
    // 填写验证码
    await page.locator('input[type="text"]').nth(2).fill("12345678");
    await page.locator('.login-form button[type="button"]').first().click();

    // 等待登录成功
    await page.waitForURL(/\/(home|dashboard|index)/, { timeout: 30000 });

    // 点击用户头像或用户菜单
    const userMenu = page.locator(
      '.user-info, [class*="user"], [class*="avatar"]'
    );
    await userMenu.click();

    // 点击登出按钮
    const logoutButton = page.getByRole("button", {
      name: /退出|登出|Logout/i,
    });
    await logoutButton.click({ timeout: 15000 });

    // 等待返回登录页
    await page.waitForURL(/\/login/, { timeout: 30000 });

    // 验证已回到登录页面
    await expect(page.locator('input[type="text"]').first()).toBeVisible();
  });
});
