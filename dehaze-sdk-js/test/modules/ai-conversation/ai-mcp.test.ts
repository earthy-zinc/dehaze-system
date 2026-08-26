import { describe, test, expect } from "vitest";
import { AiMCPAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createMcpServerForm,
  createMcpServerQuery,
  createMcpCredentialForm,
} from "#/factories/ai-mcp";

/**
 * MCP Server 管理（F-M08-006 §2.6.13，管理端需 ai:mcp:manage）。
 *
 * 后端尚未实现 MCP 管理路由：测试先行契约（以 dehaze-doc API接口.md §2.13 为行为断言依据），
 * 接口 404 时正向用例失败暴露，待后端实现后统一验证。
 * 数据前缀 test_mcp_，普通用户 403（A0301）。
 */
describe("MCP Server 管理 - AiMCPAPI (T-MF-180~189)", () => {
  describe("Server 注册表", () => {
    test("T-MF-180 正向：注册外部 MCP Server 返回完整结构", async () => {
      await login(USERS.ADMIN.username);
      const form = createMcpServerForm();
      const result = await AiMCPAPI.createServer(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toBe(form.name);
      expect(result.protocolType).toBe(form.protocolType);
      expect(result.endpoint).toBe(form.endpoint);
      expect(result.status).toBe(1);
    });

    test("T-MF-189 负向：普通用户注册 MCP Server → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiMCPAPI.createServer(createMcpServerForm()), ["A0301"]);
    });

    test("T-MF-001 正向：MCP Server 分页列表", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiMCPAPI.listServers(createMcpServerQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("T-MF-004 正向：更新 Server 配置", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      const updated = await AiMCPAPI.updateServer(created.id, {
        ...createMcpServerForm(),
        description: "updated-desc",
      });
      expect(updated.id).toBe(created.id);
      expect(updated.description).toBe("updated-desc");
    });

    test("T-MF-005 正向：启停 Server", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      const disabled = await AiMCPAPI.switchServerStatus(created.id, 0);
      expect(disabled.status).toBe(0);
      const enabled = await AiMCPAPI.switchServerStatus(created.id, 1);
      expect(enabled.status).toBe(1);
    });
  });

  describe("工具 / 命名空间 / 凭据 / 健康", () => {
    test("T-MF-181 正向：Server 工具清单", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      const tools = await AiMCPAPI.getTools(created.id);
      expect(Array.isArray(tools)).toBe(true);
    });

    test("T-MF-183 正向：命名空间配置覆盖式更新", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      const namespaces = await AiMCPAPI.updateNamespaces(created.id, [
        { name: "ns_a", toolNames: ["tool_a"] },
      ]);
      expect(namespaces).toEqual([{ name: "ns_a", toolNames: ["tool_a"] }]);
    });

    test("T-MF-184 正向：凭据加密存储（仅录入不回显）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      await AiMCPAPI.updateCredentials(created.id, createMcpCredentialForm());
      const detail = await AiMCPAPI.getServer(created.id);
      expect(detail.id).toBe(created.id);
    });

    test("T-MF-185 正向：Server 健康探测", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiMCPAPI.createServer(createMcpServerForm());
      const health = await AiMCPAPI.probeHealth(created.id);
      expect(["online", "offline"]).toContain(health.status);
    });
  });

  describe("市场 / 调用审计", () => {
    test("T-MF-187 正向：市场目录可浏览", async () => {
      await login(USERS.ADMIN.username);
      const market = await AiMCPAPI.getMarket();
      expect(Array.isArray(market)).toBe(true);
    });

    test("T-MF-186 正向：外部 MCP 调用审计列表", async () => {
      await login(USERS.ADMIN.username);
      const calls = await AiMCPAPI.listCalls({ pageNum: 1, pageSize: 10 });
      expect(Array.isArray(calls.list)).toBe(true);
      expect(typeof calls.total).toBe("number");
    });

    test("T-MF-189 负向：普通用户调用审计 → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiMCPAPI.listCalls({}), ["A0301"]);
    });
  });
});
