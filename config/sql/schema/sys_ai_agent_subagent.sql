-- ============================================================
-- 表名: sys_ai_agent_subagent
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- Agent-Subagent 关联表（自引用）。一个 Agent 可关联多个子 Agent，主 Agent 通过 deepagents
--   内置的 task 工具调用子 Agent 执行隔离子任务（独立上下文窗口，仅返回结果摘要）。
-- Team 团队成员关系也复用本表：Team Lead 为 parent_agent_id，Team Member 为 subagent_agent_id。
-- 触发描述复用子 Agent 自身的 sys_ai_agent.description 字段（单一信息源，对齐 deepagents
--   SubAgent.description 和 Claude Code subagent frontmatter description 的语义——都是子 Agent
--   自身属性，而非按调用关系区分的属性）。
-- priority 用于多个子 Agent 均可处理同一任务时的选择排序（数字越小越优先）。
-- endpoint_id 区分本地与远程子 Agent：NULL 为本地子 Agent（进程内 task 工具），非 NULL 指向
--   sys_ai_agent_endpoint 的外部 A2A Agent（走 A2A 客户端，见 §5.4）。
-- 远程子 Agent 须在本地 sys_ai_agent 建立影子记录（is_subagent=1），subagent_agent_id 指向影子记录，
--   endpoint_id 指向外部端点；运行时以 endpoint_id 非空判定走 A2A 客户端。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_subagent`;
CREATE TABLE `sys_ai_agent_subagent`
(
    `parent_agent_id`    bigint                                                        NOT NULL COMMENT '父Agent ID(关联sys_ai_agent.id,主Agent或Team Lead)',
    `subagent_agent_id`  bigint                                                        NOT NULL COMMENT '子Agent ID(关联sys_ai_agent.id,被调用的子Agent/Team Member;远程A2A子Agent为本地影子记录)',
    `endpoint_id`        bigint                                                        NULL DEFAULT NULL COMMENT '外部A2A端点ID(关联sys_ai_agent_endpoint.id;NULL为本地子Agent走task工具,非NULL为远程A2A子Agent走A2A客户端)',
    `priority`           int                                                           NOT NULL DEFAULT 0 COMMENT '优先级(数字越小越优先,多个子Agent均可处理同一任务时按此排序)',
    `create_time`        datetime                                                      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`parent_agent_id`, `subagent_agent_id`) USING BTREE,
    INDEX `idx_subagent` (`subagent_agent_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'Agent-Subagent关联表'
  ROW_FORMAT = DYNAMIC;
