-- ============================================================
-- 表名: sys_ai_agent_thought
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 推理过程表，记录 Agent 多步推理中的每个步骤（思考→工具调用→观察）。
-- 对齐 Dify MessageAgentThought 和 OpenAI RunStep 的设计，推理过程独立成表而非塞入消息 content。
-- message_id 关联触发推理的 assistant 消息，conversation_id 冗余便于按会话查询。
-- position 为步骤序号（从1开始），同一消息内按 position 排序还原推理链路。
-- thought 存 LLM 的思考内容（为什么选这个工具、预期效果）。
-- tool 存工具名称，tool_input(JSON) 存工具入参，observation 存工具返回摘要。
-- status 标识步骤结果：1:成功/2:失败/3:跳过，latency_ms 记录工具调用耗时，error 记录失败原因。
-- 推理记录为只追加，不使用逻辑删除（类似日志），过期数据通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_thought`;
CREATE TABLE `sys_ai_agent_thought`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `message_id`      bigint                                                         NOT NULL COMMENT '关联消息ID(触发推理的assistant消息)',
    `conversation_id` bigint                                                         NOT NULL COMMENT '会话ID(冗余，便于按会话查询推理链路)',
    `position`        int                                                            NOT NULL COMMENT '步骤序号(从1开始，同一消息内排序)',
    `thought`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT 'LLM思考内容(为什么选这个工具;预期效果)',
    `tool`            varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '工具名称(MCP tool标识)',
    `tool_input`      json                                                           NULL COMMENT '工具输入参数(JSON)',
    `observation`     TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '工具返回摘要(结构化摘要，不含大体积原始数据)',
    `status`          tinyint                                                        NOT NULL DEFAULT 1 COMMENT '步骤状态(1:成功;2:失败;3:跳过)',
    `latency_ms`      int                                                            NULL DEFAULT 0 COMMENT '工具调用耗时(毫秒)',
    `error`           TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '失败原因(status=2时填充)',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_message_position` (`message_id`, `position`) USING BTREE,
    INDEX `idx_conversation` (`conversation_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI推理过程表'
  ROW_FORMAT = DYNAMIC;
