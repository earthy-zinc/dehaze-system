import { expect, test } from "@playwright/test";

const PROVIDERS_PATH = "/#/admin/ai-models/providers";

/** 供应商列表接口的商品表（含 protocolType），用于数据驱动断言 */
async function fetchProviders(context: {
  request: { get: (url: string) => Promise<any> };
}) {
  const response = await context.request.get(
    "/api/v1/ai/providers?pageNum=1&pageSize=100"
  );
  const body = await response.json();
  expect(body.code, "供应商列表接口未返回成功码").toBe("00000");
  return body.data.list as { providerCode: string; protocolType: string }[];
}

/**
 * 供应商列表位于第二张卡片（第一张是健康看板表格），两张表都用 .el-table__row，
 * 因此所有行定位都必须按卡片收敛，否则会误命中看板行。
 */
function providerTable(page: import("@playwright/test").Page) {
  return page.locator(".el-card").filter({ hasText: "新增供应商" });
}

async function openProvidersPage(page: import("@playwright/test").Page) {
  await page.goto(PROVIDERS_PATH);
  await page.locator(".loader").waitFor({ state: "detached", timeout: 15000 });
  await expect(
    providerTable(page).locator(".el-table__row").first()
  ).toBeVisible();
}

test.describe("AI 供应商管理", () => {
  test("健康看板：近 24h 无调用的供应商指标显示为 -（不落成 100%/0ms）", async ({
    page,
    context,
  }) => {
    const response = await context.request.get("/api/v1/ai/usage/stats");
    const body = await response.json();
    const providerHealth = body.data.providerHealth as {
      providerName: string;
      callCount: number;
      successRate?: number | null;
      rate429?: number | null;
      p95LatencyMs?: number | null;
    }[];
    const withoutCalls = providerHealth.filter((p) => p.callCount === 0);
    expect(
      withoutCalls.length,
      "环境中没有零调用供应商，无法验证无数据展示"
    ).toBeGreaterThan(0);

    await openProvidersPage(page);
    const board = page
      .locator(".el-card")
      .filter({ hasText: "供应商健康看板" });
    await expect(board.locator(".el-table__row").first()).toBeVisible();

    for (const provider of withoutCalls) {
      // 契约层：无样本的比率/P95 必须无值（python 省略 null 字段，java/go 给 null），
      // 后端用 1.0/0 兜底会把「无数据」显示成真实指标
      expect(
        {
          successRate: provider.successRate ?? null,
          rate429: provider.rate429 ?? null,
          p95LatencyMs: provider.p95LatencyMs ?? null,
        },
        `供应商「${provider.providerName}」零调用却在返回真实指标`
      ).toEqual({ successRate: null, rate429: null, p95LatencyMs: null });

      // 展示层：成功率 / 限流(429) / P95延迟 三列都显示 -
      const cells = await board
        .locator(".el-table__row")
        .filter({ hasText: provider.providerName })
        .first()
        .locator("td")
        .allInnerTexts();
      expect(
        {
          成功率: cells[3]?.trim(),
          限流: cells[4]?.trim(),
          P95延迟: cells[5]?.trim(),
        },
        `供应商「${provider.providerName}」无数据列未显示 -`
      ).toEqual({ 成功率: "-", 限流: "-", P95延迟: "-" });
    }
  });

  test("删除确认框按钮为中文", async ({ page }) => {
    await openProvidersPage(page);
    const rows = providerTable(page).locator(".el-table__row");
    const before = await rows.count();

    await rows.first().getByRole("button", { name: "删除" }).click();
    const messageBox = page.locator(".el-message-box");
    await expect(messageBox).toBeVisible();
    // 命令式弹窗（ElMessageBox）不套用应用内 el-config-provider 的 locale 时会回落成 Cancel/OK
    await expect(
      messageBox.locator(".el-message-box__btns button").first()
    ).toHaveText("取消");
    await expect(
      messageBox.locator(".el-message-box__btns button").last()
    ).toHaveText("确定");

    await messageBox.locator(".el-message-box__btns button").first().click();
    await expect(messageBox).toBeHidden();
    await expect(rows).toHaveCount(before);
  });

  test("配置抽屉的协议类型下拉覆盖值域与已有协议族", async ({
    page,
    context,
  }) => {
    // 协议类型是后端分派依据（openai_audio/dashscope_realtime/volc_realtime→语音、openai_image→图像、
    // anthropic→Claude 原生、cohere→embedding），下拉漏项会让既有供应商显示为裸文本，保存即把协议改写为其它值
    const providers = await fetchProviders(context);
    const byProtocol = new Map<string, string>();
    for (const p of providers) {
      if (!byProtocol.has(p.protocolType)) {
        byProtocol.set(p.protocolType, p.providerCode);
      }
    }
    const unusual = [...byProtocol.keys()].filter(
      (t) => !["openai_compat", "anthropic"].includes(t)
    );
    expect(
      unusual.length,
      "环境中缺少非 openai_compat/anthropic 的供应商，无法覆盖协议选项"
    ).toBeGreaterThan(0);

    await openProvidersPage(page);

    let domainChecked = false;
    for (const [protocolType, providerCode] of byProtocol) {
      const row = providerTable(page)
        .locator(".el-table__row")
        .filter({ hasText: providerCode })
        .first();
      await row.getByRole("button", { name: "配置", exact: true }).click();

      // 布局里常驻一个隐藏的「项目配置」抽屉，必须收敛到当前可见的那个
      const drawer = page.locator(".el-drawer:visible");
      await expect(drawer).toBeVisible();
      const protocolItem = drawer
        .locator(".el-form-item")
        .filter({ hasText: "协议类型" });
      // 当前值必须被下拉如实呈现
      await expect(protocolItem).toContainText(protocolType);

      // 展开下拉后，当前值必须是可选项之一
      await protocolItem.locator(".el-select").click();
      const options = page.locator(
        ".el-select-dropdown:visible .el-select-dropdown__item"
      );
      await expect(
        options.filter({ hasText: protocolType }).first(),
        `协议类型下拉缺少选项「${protocolType}」（供应商 ${providerCode}）`
      ).toBeVisible();

      if (!domainChecked) {
        // 值域声明处：`sys_ai_provider.protocol_type` 列注释 与 需求规格「协议类型」；
        // 缺项会让管理员无法配置该类供应商（如百炼/火山实时语音）
        const protocolDomain = [
          "openai_compat",
          "anthropic",
          "openai_image",
          "openai_audio",
          "cohere",
          "dashscope_realtime",
          "volc_realtime",
          "dashscope_async",
        ];
        const offered = (await options.allInnerTexts()).map((text) =>
          text.trim()
        );
        expect(
          protocolDomain.filter((type) => !offered.includes(type)),
          "协议类型下拉缺少后端支持的协议族"
        ).toEqual([]);
        domainChecked = true;
      }

      await page.keyboard.press("Escape");
      await drawer.getByRole("button", { name: /取\s*消/ }).click();
      await expect(drawer).toBeHidden();
    }
  });

  test("手动连通测试失败时给出结论且不抛未捕获异常", async ({ page }) => {
    const uncaught: string[] = [];
    page.on("pageerror", (error) => uncaught.push(error.message));
    page.on("console", (message) => {
      if (
        message.type() === "error" &&
        message.text().includes("vue_error_handler")
      ) {
        uncaught.push(message.text());
      }
    });

    // 连通失败（无可用 Key / 供应商不可达）是预期结果，用固定响应把该分支逼出来
    await page.route("**/api/v1/ai/providers/*/test-connection", (route) =>
      route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({
          code: "A0503",
          msg: "该供应商没有可用的启用 API Key，无法进行连通性测试",
          traceId: "e2e",
        }),
      })
    );

    await openProvidersPage(page);

    const targetRow = providerTable(page)
      .locator(".el-table__row")
      .filter({ has: page.getByRole("button", { name: "连通测试" }) })
      .first();
    const providerName = (
      await targetRow.locator("td").nth(1).innerText()
    ).trim();
    await targetRow.getByRole("button", { name: "连通测试" }).click();

    const messageBox = page.locator(".el-message-box");
    await expect(
      messageBox,
      `连通测试失败未就地提示（供应商 ${providerName}）`
    ).toContainText("连通性测试失败");
    await expect(messageBox).toContainText("无法进行连通性测试");
    await messageBox.locator(".el-message-box__btns button").first().click();
    await expect(messageBox).toBeHidden();

    expect(uncaught, `出现了未捕获异常：\n${uncaught.join("\n")}`).toEqual([]);
  });
});
