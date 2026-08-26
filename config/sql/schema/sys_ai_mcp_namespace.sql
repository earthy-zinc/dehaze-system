-- ============================================================
-- 表名: sys_ai_mcp_namespace
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 外部 MCP Server 的命名空间（工具分组）配置。Agent 按命名空间关联工具实现最小权限。
-- namespace 为分组标识（对齐 SDK McpNamespaceVO.name），tool_names 存储该分组下的工具名数组
-- （JSON），避免一对多拆表，便于覆盖式整体更新（PUT 时删旧插新）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_mcp_namespace`;
CREATE TABLE `sys_ai_mcp_namespace`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `server_id`   bigint                                                          NOT NULL COMMENT '关联Server ID(关联sys_ai_mcp_server.id)',
    `namespace`   varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '命名空间标识(工具分组,对齐McpNamespaceVO.name)',
    `tool_names`  json                                                            NULL DEFAULT NULL COMMENT '分组内工具名数组(JSON)',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_server_id` (`server_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '外部MCP Server命名空间配置表'
  ROW_FORMAT = DYNAMIC;
