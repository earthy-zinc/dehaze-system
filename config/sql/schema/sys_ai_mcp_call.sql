-- ============================================================
-- 表名: sys_ai_mcp_call
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 外部 MCP 工具调用审计表（只追加，不逻辑删除）。记录谁/何时/调用了什么工具/结果/耗时，
-- 支撑调用治理与对账。request/response 记录调用载荷与响应（response 可能较大，用 TEXT）。
-- result 区分成功/失败(success/failure)；status 为调用状态码(0失败/1成功)。
-- server_name 冗余存储，避免审计时关联已被软删的 server。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_mcp_call`;
CREATE TABLE `sys_ai_mcp_call`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`     bigint                                                          NULL DEFAULT NULL COMMENT '调用用户ID(关联sys_user.id,NULL表示系统调用)',
    `server_id`   bigint                                                          NOT NULL COMMENT '关联Server ID(关联sys_ai_mcp_server.id)',
    `server_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT 'Server名称(冗余快照)',
    `tool_name`   varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '被调用的工具名',
    `request`     json                                                            NULL DEFAULT NULL COMMENT '调用载荷(JSON)',
    `response`    text                                                            NULL COMMENT '响应结果(JSON文本)',
    `status`      tinyint                                                         NOT NULL DEFAULT 0 COMMENT '调用状态(0:失败;1:成功)',
    `result`      varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'success' COMMENT '调用结果(success;failure)',
    `latency_ms`  int                                                             NULL DEFAULT NULL COMMENT '调用耗时(毫秒)',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_server_id` (`server_id`) USING BTREE,
    INDEX `idx_user_create_time` (`user_id`, `create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '外部MCP工具调用审计表(只追加)'
  ROW_FORMAT = DYNAMIC;
