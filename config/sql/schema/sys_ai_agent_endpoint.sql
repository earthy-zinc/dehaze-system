-- ============================================================
-- 表名: sys_ai_agent_endpoint
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 外部 A2A Agent 端点注册表。平台作为 A2A 客户端调用外部 Agent 时，在此注册外部端点。
-- 注册时拉取并缓存 Agent Card（agent_card），凭证经 AES 加密存储（credential），明文不落库。
-- auth_type 遵循 A2A Agent Card securitySchemes 声明的方案（OpenAPI 3.2 五种类型）：
--   apiKey / http / oauth2 / openIdConnect / mutualTLS。
-- 子 Agent 关联（sys_ai_agent_subagent.endpoint_id）指向本表，区分本地/远程子 Agent。
-- 配置类表，使用逻辑删除；唯一键 base_url 不含 deleted（类别①，upsert 复活）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_endpoint`;
CREATE TABLE `sys_ai_agent_endpoint`
(
    `id`             bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`           varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '端点名称',
    `agent_card_url` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT 'Agent Card地址(发现端点,如 https://host/.well-known/agent.json)',
    `base_url`       varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT 'A2A端点地址(如 https://host/a2a)',
    `auth_type`      varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'http' COMMENT '认证方式(apiKey;http;oauth2;openIdConnect;mutualTLS,遵循Agent Card securitySchemes声明)',
    `credential`     varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '凭证密文(AES加密后base64编码,运行时解密按声明方案注入请求头)',
    `agent_card`     json                                                            NULL COMMENT '缓存的Agent Card JSON(注册时拉取,作为发现依据)',
    `status`         tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`        tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`      bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`    datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_base_url` (`base_url`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI外部A2A端点注册表'
  ROW_FORMAT = DYNAMIC;
