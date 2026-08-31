-- ============================================================
-- 表名: sys_ai_trace
-- 模块: 核心模块-AI对话（可观测性 F-M08-013）
-- ============================================================
-- 设计思路:
-- 对话过程链汇总记录：每次助手回复（成功/失败/中断/超时）一条记录。
-- 推理步骤明细复用 sys_ai_agent_thought（thought/tool/observation/status/latencyMs），本表不复制；
-- context_snapshot(JSON) 存上下文构成快照（系统提示/历史/记忆/检索/工具清单及各占比、
-- 压缩/截断事件），生成时不可变落盘，支撑"AI 当时看到什么"可审计可回放；
-- trace_id 复用日志链路 trace_id（唯一索引保证幂等），异常详情可从日志按 trace_id 关联。
-- 日志/历史类表，只追加记录，不删除、不使用逻辑删除；保留 180 天由定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_trace`;
CREATE TABLE `sys_ai_trace`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `trace_id`          varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '过程链ID(复用日志链路trace_id，全链路串联)',
    `conversation_id`   bigint                                                         NOT NULL COMMENT '所属会话ID(关联sys_ai_conversation.id)',
    `message_id`        bigint                                                         NULL DEFAULT NULL COMMENT '关联助手回复消息ID(关联sys_ai_message.id)',
    `agent_code`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '执行智能体编码',
    `trace_type`        varchar(32)                                                    NOT NULL DEFAULT 'conversation' COMMENT '过程链类型(conversation主对话; summary会话摘要压缩; memory_extraction记忆提取; suggestion类似问题推荐; step_summary步骤摘要)',
    `model`             varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '实际使用模型标识',
    `status`            tinyint                                                        NOT NULL DEFAULT 1 COMMENT '执行状态(1:成功;2:失败;3:中断;4:超时)',
    `error_type`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '失败类型(工具失败/模型异常/超时/配额拒绝)',
    `duration_ms`       int                                                            NOT NULL DEFAULT 0 COMMENT '整条回复总耗时(毫秒)',
    `first_token_ms`    int                                                            NULL DEFAULT NULL COMMENT '首Token延迟(毫秒，首个输出token到达耗时)',
    `llm_call_count`    int                                                            NOT NULL DEFAULT 0 COMMENT '本次回复的LLM调用次数',
    `total_tokens`      int                                                            NOT NULL DEFAULT 0 COMMENT '总Token消耗(与计费口径一致)',
    `prompt_tokens`     int                                                            NOT NULL DEFAULT 0 COMMENT '输入Token消耗',
    `completion_tokens` int                                                            NOT NULL DEFAULT 0 COMMENT '输出Token消耗',
    `cached_tokens`     int                                                            NOT NULL DEFAULT 0 COMMENT '缓存命中Token数(与计费口径一致)',
    `step_count`        int                                                            NOT NULL DEFAULT 0 COMMENT '推理步数(防循环观测，超阈值显著标注)',
    `context_snapshot`  json                                                           NULL COMMENT '上下文构成快照JSON(系统提示/历史/记忆/检索/工具清单及各占比、压缩/截断事件)',
    `error_detail`      json                                                           NULL COMMENT '异常详情(消息+堆栈截断,失败/中断时填充)',
    `create_time`       datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_trace_id` (`trace_id`) USING BTREE,
    INDEX `idx_conversation` (`conversation_id`) USING BTREE,
    INDEX `idx_message` (`message_id`) USING BTREE,
    INDEX `idx_agent_model` (`agent_code`, `model`) USING BTREE,
    INDEX `idx_status_time` (`status`, `create_time`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话过程链汇总记录表'
  ROW_FORMAT = DYNAMIC;
