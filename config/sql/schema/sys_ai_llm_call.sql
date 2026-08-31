-- ============================================================
-- 表名: sys_ai_llm_call
-- 模块: 核心模块-AI对话（可观测性 F-M08-013）
-- ============================================================
-- 设计思路:
-- 每次 LLM 调用明细（span 级）：多步推理中每次模型调用一条记录。
-- 经 trace_id 关联过程链（sys_ai_trace），seq 串联调用链路，step_position 关联推理步骤
-- （sys_ai_agent_thought.position，模型调用不总落在 thought 步骤边界，可为空）。
-- input_snapshot(JSON) 存本轮输入构成（system/消息按角色计数/tools/用户信息）；
-- output_snapshot(JSON) 存输出摘要（文本截断 + tool_calls 参数），不存完整输出正文；
-- cached_tokens 存缓存命中 token（LlmClient 流式结束 usage 返回），未提供置 0。
-- 日志/历史类表，只追加记录，不删除、不使用逻辑删除；保留 180 天由定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_llm_call`;
CREATE TABLE `sys_ai_llm_call`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `trace_id`          varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '关联过程链ID(关联sys_ai_trace.trace_id)',
    `seq`               int                                                            NOT NULL COMMENT '调用序号(1起递增，贯穿推理步骤链路)',
    `step_position`     int                                                            NULL DEFAULT NULL COMMENT '关联推理步骤序号(关联sys_ai_agent_thought.position，可为空)',
    `model`             varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '本次调用模型(多步推理中可能切换模型)',
    `status`            tinyint                                                        NOT NULL DEFAULT 1 COMMENT '调用状态(1:成功;2:失败;3:超时)',
    `error_type`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '失败类型',
    `duration_ms`       int                                                            NOT NULL DEFAULT 0 COMMENT '本次调用总耗时(毫秒)',
    `first_token_ms`    int                                                            NULL DEFAULT NULL COMMENT '本次调用首Token延迟(毫秒，首个输出token到达耗时)',
    `prompt_tokens`     int                                                            NOT NULL DEFAULT 0 COMMENT '输入Token消耗',
    `completion_tokens` int                                                            NOT NULL DEFAULT 0 COMMENT '输出Token消耗',
    `cached_tokens`     int                                                            NOT NULL DEFAULT 0 COMMENT '缓存命中Token数(未提供缓存统计的模型置0)',
    `tool_call`         json                                                           NULL COMMENT '工具调用信息JSON(has_tool_call/tool_name/args_summary)',
    `input_snapshot`    json                                                           NULL COMMENT '本次调用输入构成JSON(system/消息按角色计数/tools/用户信息)',
    `output_snapshot`   json                                                           NULL COMMENT '本次调用输出摘要JSON(文本截断 + tool_calls参数)',
    `attempts`          json                                                           NULL COMMENT '物理调用尝试明细JSON(逐Key/逐路由: provider_id/key_id/model/status/error_code/latency_ms)',
    `create_time`       datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_trace_seq` (`trace_id`, `seq`) USING BTREE,
    INDEX `idx_step` (`step_position`) USING BTREE,
    INDEX `idx_model_time` (`model`, `create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话每次LLM调用明细表'
  ROW_FORMAT = DYNAMIC;
