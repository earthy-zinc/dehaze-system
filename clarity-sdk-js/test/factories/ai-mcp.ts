import { pageQuery } from "./common";
import { uniqueName } from "./common";
import type { McpCredentialForm, McpServerForm, McpServerQuery } from "../../src/api/ai-mcp/model";

/** MCP Server 注册表单工厂（名称前缀 test_mcp_ 便于清理） */
export const createMcpServerForm = (overrides?: Partial<McpServerForm>): McpServerForm => ({
  name: uniqueName("test_mcp"),
  description: "MCP Server 管理契约测试",
  protocolType: "streamable-http",
  endpoint: "https://example.com/mcp",
  authType: "api_key",
  ...overrides,
});

/** MCP Server 分页查询参数工厂 */
export const createMcpServerQuery = (overrides?: Partial<McpServerQuery>) =>
  pageQuery<McpServerQuery>({ ...overrides });

/** MCP 凭据配置表单工厂 */
export const createMcpCredentialForm = (
  overrides?: Partial<McpCredentialForm>
): McpCredentialForm => ({
  apiKey: "test_secret_key",
  ...overrides,
});
