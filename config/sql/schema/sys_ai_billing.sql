-- ============================================================
-- 表名: sys_ai_billing
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- AI 计费记录表，记录每次 LLM 对话的 Token 消耗明细和积分扣减。
-- conversation_id/message_id 关联 AI 对话模块的会话和消息（bigint 类型，与 sys_ai_conversation/sys_ai_message 主键一致）。
-- model 记录实际使用的模型标识（降级场景记录降级模型，用于按模型维度统计成本）。
-- actual_model 记录用户原本选择的模型标识（降级场景区分用户意图和实际执行结果，NULL 表示未降级）。
-- input_tokens 含缓存命中部分，cached_input_tokens 为其中缓存命中的子集。
-- credits 为按实际使用模型计费比例换算后的积分，credits_saved 为缓存命中节省的积分。
-- bill_type 标识计费类型（message:消息LLM调用;tool_llm:工具LLM推理如ReAct observe;subagent:子智能体）。
-- tool_credits 单独记录工具调用中的 LLM Token 消耗（工具编排可能额外调 LLM，如观察结果）。
-- pre_deduct 标识预扣减积分数（发送消息前预估扣减，完成后实扣 difference）。
-- 保留原始 Token 数用于成本分析与模型对比，credits 用于配额扣减。
-- 计费记录为只追加，不使用逻辑删除（财务记录不可删除），过期数据通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_billing`;
CREATE TABLE `sys_ai_billing`
(
    `id`                  bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`             bigint                                                         NOT NULL COMMENT '用户ID',
    `conversation_id`     bigint                                                         NOT NULL COMMENT '会话ID(关联sys_ai_conversation.id)',
    `message_id`          bigint                                                         NOT NULL COMMENT '消息ID(关联sys_ai_message.id)',
    `model`               varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '实际使用的模型标识(降级场景记录降级模型)',
    `actual_model`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户原选模型标识(降级场景区分用户意图，NULL表示未降级)',
    `bill_type`           varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'message' COMMENT '计费类型(message:消息LLM调用;tool_llm:工具LLM推理;subagent:子智能体)',
    `input_tokens`        bigint                                                         NOT NULL DEFAULT 0 COMMENT '输入Token数(含缓存命中部分)',
    `cached_input_tokens` bigint                                                         NOT NULL DEFAULT 0 COMMENT '其中缓存命中的输入Token数',
    `output_tokens`       bigint                                                         NOT NULL DEFAULT 0 COMMENT '输出Token数',
    `credits`             bigint                                                         NOT NULL DEFAULT 0 COMMENT '消耗积分数(按实际使用模型计费比例换算后)',
    `credits_saved`       bigint                                                         NOT NULL DEFAULT 0 COMMENT '缓存命中节省的积分数(未缓存时应扣减积分-实扣积分)',
    `tool_credits`        bigint                                                         NOT NULL DEFAULT 0 COMMENT '工具调用中额外消耗的LLM Token积分(如ReAct observe步骤)',
    `quota_consumed`      bigint                                                         NOT NULL DEFAULT 0 COMMENT '实际扣减的配额(credits-预扣退还差额)',
    `pre_deduct`          bigint                                                         NOT NULL DEFAULT 0 COMMENT '预扣减积分数(发送消息前预估扣减)',
    `create_by`           bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`           bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`         datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_time` (`user_id`, `create_time`) USING BTREE,
    INDEX `idx_conversation` (`conversation_id`) USING BTREE,
    INDEX `idx_message` (`message_id`) USING BTREE,
    INDEX `idx_bill_type` (`bill_type`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI计费记录表'
  ROW_FORMAT = DYNAMIC;
