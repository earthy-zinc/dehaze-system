-- ============================================================
-- 迁移: AI 对话 v2 契约 —— sys_ai_message 破坏性重建
-- 日期: 2026-10-21
-- 类型: DESTRUCTIVE（不可回滚）
-- ============================================================
-- ⚠️ 本迁移会 DROP TABLE `sys_ai_message`，该表存量数据全部丢弃，不做迁移、不做回填。
--    依据《AI对话/架构收敛方案》：不兼容旧数据、旧协议，上线前按 scripts/ai-chat-clear-data.sql
--    清空 AI 对话域数据，切换后由新实现从零写入。
--
-- 执行前检查:
--   1. 确认目标库与环境（dev/pre/prod），核对 MYSQL_DATABASE 与连接串；
--   2. 已停止旧版本写入（旧代码仍按 content/tool_calls/status tinyint 写本表，会直接报错）；
--   3. 已按运维要求备份，或确认存量数据允许丢弃；
--   4. 本表被 sys_ai_billing / sys_ai_trace / sys_ai_artifact / sys_ai_agent_thought /
--      sys_ai_message_feedback 以 message_id 引用（无外键约束），DROP 不会失败。
--      scripts/ai-chat-clear-data.sql 会清空 artifact / agent_thought / feedback；
--      billing / trace / llm_call 保留不动，其中 trace 的旧 conversation_id / message_id
--      会成为悬空引用，由运维单独评估是否清理或保留。
--
-- 结构变更:
--   删除: content、tool_calls、tool_call_id、metadata、original_content
--   新增: blocks(NOT NULL)、plan、pending_interrupt、suggestions、sub_agent_usage
--   变更: status 由 tinyint(1:流式/2:完成/3:失败/4:取消) 改为
--         varchar(16) NOT NULL DEFAULT 'running'（running/completed/failed/canceled/awaiting_input）
--   保留: id/conversation_id/parent_message_id/role/model/error/input_tokens/output_tokens/
--         cached_input_tokens/credits/task_id/used_memory_ids/edited/deleted/create_by/update_by/
--         create_time/update_time 及原有索引
--
-- 执行后状态与 config/sql/schema/sys_ai_message.sql 完全一致（两处须同步修改）。
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
