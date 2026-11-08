-- ============================================================
-- 表名: sys_ai_message
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 对话消息表，记录会话中的每一条消息（system/user/assistant/tool）。
-- 消息只有一种形状：正文、思考、工具调用统一由 blocks(JSON) 的 Block[] 承载，
--   kind 取 text/thinking/tool_use，seq 表示同一消息内同 kind 块的先后；
--   渲染按 kind 分区（思考、工具、正文各归各位），分区内按 seq 保序，不做跨 kind 交错。
-- plan(JSON) 持久化执行计划（Plan:phase/status/tasks/revisions/awaitingApproval），
--   修订记录带 actor 与时间；plan 挂在消息上，刷新后读到同一份计划。
-- pending_interrupt(JSON) 持久化待恢复中断（Interrupt:含 expiresAt/actionable），
--   resume 从持久状态恢复；超过 expiresAt 由 actionable=false 明确标注不可续。
-- suggestions(JSON) 存推荐追问列表（string[]）；产物与反馈是独立资源
--   （sys_ai_artifact / sys_ai_message_feedback），消息只存 artifact_ids 引用，不存正文。
-- role 枚举 system/user/assistant/tool，覆盖 OpenAI/Claude 消息角色体系。
-- status 只表达本轮通道，等待确认由 awaiting_input 与 pending_interrupt 共同表达：
--   running:运行中;completed:已完成;failed:失败;canceled:已取消;awaiting_input:等待用户输入。
-- 流式断线重连依赖 status 与 blocks 的落库形态，不额外存增量帧。
-- model 记录本条消息实际使用的模型（一次会话可能切换模型）；error 记录失败原因。
-- input_tokens/output_tokens/cached_input_tokens 记录 token 消耗，credits 记录换算后积分。
-- task_id 关联 sys_task，用于异步等待场景（如图像处理完成后恢复推理）。
-- used_memory_ids(JSON) 记录本次注入引用的记忆ID列表，支撑注入可见性。
-- edited 标识消息是否被编辑过；parent_message_id 支持分支对话（重新生成、编辑后重发）。
-- 消息支持逻辑删除，无业务唯一键，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_message`;
CREATE TABLE `sys_ai_message`
(
    `id`                   bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `conversation_id`      bigint                                                          NOT NULL COMMENT '会话ID(关联sys_ai_conversation.id)',
    `parent_message_id`    bigint                                                          NULL DEFAULT NULL COMMENT '父消息ID(支持分支对话:重新生成/编辑后重发)',
    `role`                 varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '消息角色(system;user;assistant;tool)',
    `status`               varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'running' COMMENT '消息状态(running:运行中;completed:已完成;failed:失败;canceled:已取消;awaiting_input:等待用户输入)',
    `blocks`               json                                                            NOT NULL COMMENT '内容块列表(Block[]:id/kind/seq/text/durationMs/tool，kind取text;thinking;tool_use)',
    `plan`                 json                                                            NULL COMMENT '执行计划(Plan:phase/status/tasks/revisions/awaitingApproval，修订带actor与时间)',
    `pending_interrupt`    json                                                            NULL COMMENT '待恢复中断(Interrupt:含expiresAt/actionable，resume从持久状态恢复)',
    `suggestions`          json                                                            NULL COMMENT '推荐追问列表(string[])',
    `model`                varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '本条消息使用的模型标识(一次会话可能切换模型)',
    `error`                TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '错误信息(status=failed时填充)',
    `input_tokens`         int                                                             NOT NULL DEFAULT 0 COMMENT '输入Token数(含缓存命中部分)',
    `output_tokens`        int                                                             NOT NULL DEFAULT 0 COMMENT '输出Token数',
    `cached_input_tokens`  int                                                             NOT NULL DEFAULT 0 COMMENT '其中缓存命中的输入Token数',
    `credits`              bigint                                                          NOT NULL DEFAULT 0 COMMENT '消耗积分数(按模型计费比例换算后)',
    `task_id`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '关联异步任务ID(等待图像处理等异步任务时使用)',
    `used_memory_ids`      json                                                            NULL COMMENT '本次注入引用的记忆ID列表(JSON数组,注入可见性)',
    `sub_agent_usage`      json                                                            NULL COMMENT '子智能体分项用量(SubAgentUsage[]，与 usage.subAgents 同形)',
    `edited`               tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否已编辑(0:否;1:是)',
    `deleted`              bigint                                                          NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)',
    `create_by`            bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`            bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`          datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`          datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_conversation_time` (`conversation_id`, `create_time`) USING BTREE,
    INDEX `idx_parent` (`parent_message_id`) USING BTREE,
    INDEX `idx_task` (`task_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话消息表'
  ROW_FORMAT = DYNAMIC;
