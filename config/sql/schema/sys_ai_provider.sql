-- ============================================================
-- 表名: sys_ai_provider
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 模型供应商配置表，管理平台接入的 LLM 供应商（OpenAI/Anthropic/DeepSeek 等）。
-- provider_code 为业务唯一键（如 openai/anthropic/deepseek），删除后不可复用（类别②）。
-- api_base_url 为供应商 API 端点，支持代理地址（如自建 OpenAI 代理）。
-- protocol_type 决定请求/响应协议：openai_compat（OpenAI 兼容）或 anthropic（Claude 原生）。
-- auth_type 决定 HTTP 认证头格式：bearer（Authorization: Bearer）、x-api-key 或 custom（自定义请求头，头名在 default_headers 中配置）。
-- default_headers 存储供应商特有请求头（如 anthropic-version），JSON 格式。
-- sort_order 为排序序号，管理员可调整供应商展示顺序。
-- health_check_enabled 为健康检查开关（默认开启），关闭后该供应商不参与熔断判定（健康状态运行时聚合于 Redis，不落库）。
-- remark 为运维备注（账号归属、合同号、商务信息），不参与逻辑。
-- 供应商下的 API Key 管理见 sys_ai_provider_key 表。
-- 配置类表，使用逻辑删除；provider_code 为业务引用键，删除后不可复用（类别②）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_provider`;
CREATE TABLE `sys_ai_provider`
(
    `id`              bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `provider_code`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '供应商编码(openai;anthropic;deepseek;zhipu;qwen;custom)',
    `display_name`    varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '显示名称(如OpenAI;Anthropic;DeepSeek)',
    `api_base_url`    varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT 'API基础地址(如https://api.openai.com/v1)',
    `protocol_type`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'openai_compat' COMMENT '协议类型(openai_compat:OpenAI兼容;anthropic:Claude原生)',
    `auth_type`       varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'bearer' COMMENT '认证方式(bearer:Authorization Bearer;x-api-key:Anthropic风格;custom:自定义请求头,头名在default_headers配置)',
    `default_headers` json                                                            NULL COMMENT '默认请求头(JSON,如{"anthropic-version":"2023-06-01"});auth_type=custom时,需含{"auth_header":"头名"}',
    `sort_order`      int                                                             NOT NULL DEFAULT 0 COMMENT '排序序号(数字越小越靠前)',
    `health_check_enabled` tinyint                                                    NOT NULL DEFAULT 1 COMMENT '健康检查开关(1:开启,参与熔断判定;0:关闭)',
    `remark`          varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL DEFAULT NULL COMMENT '运维备注(账号归属/合同号/商务信息)',
    `status`          tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`         tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`       bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_provider_code` (`provider_code`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI模型供应商配置表'
  ROW_FORMAT = DYNAMIC;
