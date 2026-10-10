import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

interface RouteNode {
  path?: string;
  component?: string;
  children?: RouteNode[];
}

/**
 * 拍平后端菜单为页面路径清单。
 *
 * 与前端 permission store 同口径：子弹 path 相对父级拼接；component 为空的按钮权限项
 * 与 Layout 目录节点不构成独立页面。清单来源是菜单接口而非硬编码，菜单增删自动跟随。
 */
function collectPagePaths(nodes: RouteNode[], parentPath = ""): string[] {
  const paths: string[] = [];
  for (const node of nodes) {
    const path = node.path ?? "";
    if (!path) {
      continue;
    }
    const fullPath = path.startsWith("/")
      ? path
      : parentPath
        ? `${parentPath.replace(/\/$/, "")}/${path}`
        : path;
    if (node.component && node.component !== "Layout") {
      paths.push(fullPath);
    }
    if (node.children?.length) {
      paths.push(...collectPagePaths(node.children, fullPath));
    }
  }
  return paths;
}

async function openRoute(page: Page, route: string): Promise<string[]> {
  const problems: string[] = [];
  const onPageError = (error: Error) =>
    problems.push(`pageerror: ${error.message}`);
  const onConsole = (message: ConsoleMessage) => {
    if (message.type() === "error") {
      problems.push(`console: ${message.text()}`);
    }
  };
  page.on("pageerror", onPageError);
  page.on("console", onConsole);

  try {
    await page.goto(`/#${route}`, { waitUntil: "domcontentloaded" });
    // 挂载点是 index.html 里的 #app，其 loader 占位在应用挂载后被替换掉
    await page
      .locator(".loader")
      .waitFor({ state: "detached", timeout: 15000 });
    const hash = new URL(page.url()).hash;
    if (/^#\/(401|403|404)/.test(hash)) {
      problems.push(`被重定向到 ${hash}`);
    }
  } catch (error) {
    problems.push(`打开失败: ${(error as Error).message.split("\n")[0]}`);
  } finally {
    page.off("pageerror", onPageError);
    page.off("console", onConsole);
  }
  return problems;
}

test.describe("路由冒烟", () => {
  test("菜单里的每个页面都能打开且无页面级报错", async ({ page, context }) => {
    test.setTimeout(15 * 60 * 1000);

    const response = await context.request.get("/api/v1/menus/routes");
    const body = await response.json();
    expect(body.code, "菜单路由接口未返回成功码").toBe("00000");

    const routes = collectPagePaths(body.data ?? []);
    expect(routes.length, "菜单里没有解析出任何页面路由").toBeGreaterThan(0);

    const failures: string[] = [];
    for (const route of routes) {
      const problems = await openRoute(page, route);
      if (problems.length > 0) {
        failures.push(`${route}\n    ${problems.join("\n    ")}`);
      }
    }

    expect(failures, `以下路由存在问题：\n  ${failures.join("\n  ")}`).toEqual(
      []
    );
  });
});
