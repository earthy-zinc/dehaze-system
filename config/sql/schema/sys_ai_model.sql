-- ============================================================
-- 表名: sys_ai_model
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 模型配置表，管理平台接入的大语言模型清单及计费比例。
-- model_id 为模型标识（如 gpt-4o、claude-3-5-sonnet），是业务引用键，唯一索引。
-- model_type 标识模型类型（chat/embedding/rerank），consumers 按类型选择模型（对话选 chat、知识库选 embedding/rerank）。
-- dimension 为 embedding 向量维度（model_type=embedding 时必填，知识库 ES 索引映射依赖，创建后不可修改）。
-- provider_id 关联 sys_ai_provider.id，标识模型所属供应商（替代原 provider 字符串字段）。
--   同一 model_id 可通过不同 provider_id 配置多行，实现"同一模型切换供应商"。
--   唯一约束为 (model_id, provider_id) 联合唯一，而非 model_id 单列唯一。
-- 用户售价（绝对单价，积分/百万 token，高峰/空闲双档，价格版本化）存 sys_ai_model_price，
--   供应商采购价（成本单价）存 sys_ai_model_cost，sys_ai_model 不再承载价格字段。
-- supports_multimodal/supports_tool_call/supports_streaming/supports_prompt_cache/supports_structured_output
--   标识模型能力，供前端模型选择和后端能力校验。
-- fallback_model_id 关联 sys_ai_model.id（主键），指定降级模型（主模型限流/宕机时自动切换），为空表示无降级；
--   用主键引用而非 model_id，避免同一 model_id 在多供应商下降级指向歧义。
-- prompt_cache_prefix_len 标识 prompt caching 稳定前缀长度，用于 prompt 结构优化（OpenAI/Anthropic 要求稳定前缀在头部）。
-- 配置类表，使用逻辑删除；model_id 为业务引用键，删除后不可复用（类别②，查重查全表）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_model`;
CREATE TABLE `sys_ai_model`
(
    `id`                   bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `provider_id`          bigint                                                          NOT NULL COMMENT '关联供应商ID(关联sys_ai_provider.id)',
    `model_id`             varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '模型标识(如gpt-4o;claude-3-5-sonnet;deepseek-chat)',
    `model_type`           varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'chat' COMMENT '模型类型(chat:对话;embedding:向量;rerank:重排)',
    `dimension`            bigint                                                          NULL DEFAULT NULL COMMENT 'embedding向量维度(model_type=embedding时必填;创建后不可改)',
    `display_name`         varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '显示名称',
    `max_context_tokens`   int                                                             NOT NULL DEFAULT 4096 COMMENT '最大上下文Token数',
    `max_output_tokens`    int                                                             NOT NULL DEFAULT 4096 COMMENT '最大输出Token数',
    `supports_multimodal`  tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否支持多模态(图片视觉/文档理解/音视频理解)(0:否;1:是)',
    `supports_tool_call`   tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否支持工具调用(0:否;1:是)',
    `supports_streaming`   tinyint                                                         NOT NULL DEFAULT 1 COMMENT '是否支持流式输出(0:否;1:是)',
    `supports_prompt_cache` tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否支持Prompt缓存(0:否;1:是,OpenAI/Anthropic缓存优化)',
    `supports_structured_output` tinyint                                                    NOT NULL DEFAULT 0 COMMENT '是否支持结构化输出(0:否;1:是,JSON Schema约束)',
    `extra_request_params` json                                                             NULL COMMENT '厂商私有请求参数(如阿里云enable_thinking/reasoning_effort)，随请求体透传，核心键不可覆盖',
    `fallback_model_id`    bigint                                                          NULL DEFAULT NULL COMMENT '降级模型ID(关联sys_ai_model.id主键,主模型限流/宕机时自动切换;用主键引用避免同model_id多供应商歧义)',
    `prompt_cache_prefix_len` int                                                           NOT NULL DEFAULT 0 COMMENT 'Prompt缓存稳定前缀长度(用于prompt结构优化，缓存命中前缀部分)',
    `image_tokens_per_image` int                                                           NOT NULL DEFAULT 0 COMMENT '多模态单图Token换算上限(0=无该规则;预扣估算按图片数×上限计入输入Token,实扣以供应商返回为准)',
    `concurrency_limit`      int                                                           NULL DEFAULT NULL COMMENT '供应商侧并发上限参考值(如deepseek-v4-flash为2500;用于平台侧限流与Key级RPM配置参考)',
    `status`               tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `vip_level`            tinyint                                                         NOT NULL DEFAULT 0 COMMENT '最低可用VIP等级(0:所有用户;1:VIP1及以上;2:VIP2及以上)',
    `deleted`              tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`            bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`            bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`          datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`          datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_model_provider` (`model_id`, `provider_id`) USING BTREE,
    INDEX `idx_provider` (`provider_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI模型配置表'
  ROW_FORMAT = DYNAMIC;
