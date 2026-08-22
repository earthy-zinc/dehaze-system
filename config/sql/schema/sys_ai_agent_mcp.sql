-- ============================================================
-- 表名: sys_ai_agent_mcp
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- Agent-MCP 命名空间关联表。一个 Agent 可关联多个 MCP 命名空间（如 image_processing、evaluation），
--   DehazeToolsBuilder 按此关联装载 mcp_lookup_tool / mcp_execute_tool 工具。
-- mcp_namespace 标识 MCP 工具分组（命名空间由 MCP 能力网关管理，本表仅记录关联关系）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_mcp`;
CREATE TABLE `sys_ai_agent_mcp`
(
    `agent_id`        bigint                                                        NOT NULL COMMENT '关联Agent ID(关联sys_ai_agent.id)',
    `mcp_namespace`  varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT 'MCP命名空间(如image_processing/evaluation)',
    `create_time`    datetime                                                      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`agent_id`, `mcp_namespace`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'Agent-MCP命名空间关联表'
  ROW_FORMAT = DYNAMIC;
