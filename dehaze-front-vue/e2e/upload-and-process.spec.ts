import { expect, test } from "@playwright/test";

test.describe("图像去雾处理流程", () => {
  test("处理页渲染模型选择与图片上传入口", async ({ page }) => {
    await page.goto("/#/presentation/dehaze");

    await expect(page.getByTestId("model-select")).toBeVisible();
    // 上传交互的 input 常驻 DOM 但视觉隐藏，用 attached 断言
    await expect(page.locator('input[type="file"]').first()).toBeAttached();
  });
});
