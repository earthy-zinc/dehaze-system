-- ============================================================
-- 表名: sys_ai_billing
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- 计费记录表，记录每次 AI 能力调用（AI 对话/工具推理/知识库注入/语音）的
-- Token/时长/字符消耗明细与积分扣减，支撑成本分析与异常审计。
-- model 记录实际使用的模型（降级场景记录降级模型），actual_model 记录用户原选模型
-- （NULL 表示未降级），二者对比可统计"用户期望成本 vs 实际成本"。
-- bill_type 区分计费类型：chat(对话回复)/tool_llm(工具推理)/kb_inject(知识库注入)/asr(语音识别)/tts(语音合成)。
-- credits 按实际模型计费比例换算（input×inputRate + cached×cachedRate + output×outputRate），
-- credits_saved 为缓存命中节省积分，tool_credits 记录工具调用额外 LLM Token 积分。
-- quota_consumed 为实际扣减配额，pre_deduct 为预扣减积分，差额用于退补对账。
-- 只追加表，不逻辑删除（财务记录不可删），过期数据通过定时任务物理清理。
-- 日志表不承载通用操作人审计（create_by/update_by），仅保留 create_time；
--   计费记录由 AI 调用系统自动产生，归属用户已由 user_id 表达。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_billing`;
CREATE TABLE `sys_ai_billing`
(
    `id`                   bigint   NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`              bigint   NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `conversation_id`      bigint   NULL DEFAULT NULL COMMENT '会话ID(关联sys_ai_conversation.id)',
    `message_id`           bigint   NULL DEFAULT NULL COMMENT '消息ID(关联sys_ai_message.id)',
    `request_id`           varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '请求唯一ID(支撑对账与异常追溯)',
    `provider_id`          bigint   NULL DEFAULT NULL COMMENT '实际供应商ID(关联sys_ai_provider.id)',
    `model`                varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '实际使用模型标识(降级场景为降级模型)',
    `actual_model`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户原选模型标识(NULL表示未降级)',
    `error_code`           varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '调用失败错误码(如429/5xx,成功为NULL)',
    `latency_ms`           int      NULL DEFAULT NULL COMMENT '调用耗时(毫秒)',
    `bill_type`            varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '计费类型(chat;tool_llm;kb_inject;asr;tts)',
    `input_tokens`         int      NOT NULL DEFAULT 0 COMMENT '输入Token数(含缓存命中部分)',
    `cached_input_tokens`  int      NOT NULL DEFAULT 0 COMMENT '其中缓存命中的输入Token数',
    `output_tokens`        int      NOT NULL DEFAULT 0 COMMENT '输出Token数',
    `credits`              int      NOT NULL DEFAULT 0 COMMENT '消耗积分数(按实际模型计费比例换算)',
    `credits_saved`        int      NOT NULL DEFAULT 0 COMMENT '缓存命中节省积分数',
    `tool_credits`         int      NULL DEFAULT NULL COMMENT '工具调用额外LLM Token积分(tool_llm类型记录)',
    `quota_consumed`       int      NOT NULL DEFAULT 0 COMMENT '实际扣减配额(credits-预扣退还差额)',
    `pre_deduct`           int      NOT NULL DEFAULT 0 COMMENT '预扣积分数',
    `create_time`          datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_create_time` (`user_id`, `create_time`) USING BTREE,
    INDEX `idx_conversation_id` (`conversation_id`) USING BTREE,
    INDEX `idx_message_id` (`message_id`) USING BTREE,
    INDEX `idx_bill_type` (`bill_type`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI计费记录表'
  ROW_FORMAT = DYNAMIC;
