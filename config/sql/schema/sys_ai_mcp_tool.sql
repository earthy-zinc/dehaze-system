-- ============================================================
-- 表名: sys_ai_mcp_tool
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 外部 MCP Server 的工具清单。注册时自动拉取工具定义，也可通过工具清单接口单独刷新。
-- input_schema 存储工具的参数 schema 概要（JSON，供 Agent 调用时构造参数）。
-- 工具随所属 Server 生命周期管理：Server 删除时级联清理本表（同表内同名覆盖更新，
-- 故不设唯一键，删除/重建由服务层按 server_id 控制）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_mcp_tool`;
CREATE TABLE `sys_ai_mcp_tool`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `server_id`   bigint                                                          NOT NULL COMMENT '关联Server ID(关联sys_ai_mcp_server.id)',
    `name`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '工具名(Server内唯一)',
    `description` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '工具描述',
    `input_schema` json                                                           NULL DEFAULT NULL COMMENT '参数schema概要(JSON)',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_server_id` (`server_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '外部MCP Server工具清单表'
  ROW_FORMAT = DYNAMIC;
