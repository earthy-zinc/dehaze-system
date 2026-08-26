-- ============================================================
-- 表名: sys_ai_mcp_server
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 外部 MCP Server 的注册中心。任何符合 MCP 规范的外部服务（stdio/streamable-http/sse）
-- 均可注册接入，工具按命名空间分组供 Agent 关联（最小权限）。
-- credentials 存储 AES 加密后的密文（JSON，仅录入/更新，不回显明文、不暴露给 LLM）。
-- health 记录最近一次健康探测结果（online/offline），探测失败不阻断管理流程。
-- tool_count 冗余工具数量，避免列表接口逐条子查询。
-- name 全局唯一，覆盖注册/更新查重（含软删历史不可复用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_mcp_server`;
CREATE TABLE `sys_ai_mcp_server`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT 'Server名称(唯一)',
    `description` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '描述',
    `protocol_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'streamable-http' COMMENT '传输协议(stdio;streamable-http;sse)',
    `endpoint`    varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '端点URL(stdio可为空)',
    `auth_type`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '鉴权方式(none;api_key;oauth2等)',
    `credentials` json                                                            NULL DEFAULT NULL COMMENT '凭据密文(JSON,AES加密后base64,仅录入/更新不回显)',
    `health`      varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '健康状态(online;offline)',
    `status`      tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `tool_count`  int                                                             NOT NULL DEFAULT 0 COMMENT '工具数量(冗余,注册/拉取时更新)',
    `deleted`     tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE KEY `uk_name` (`name`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '外部MCP Server注册表'
  ROW_FORMAT = DYNAMIC;
