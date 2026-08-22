-- ============================================================
-- 表名: sys_ai_agent_version
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- Agent 配置版本快照表，实现"草稿/已发布"分离的不可变快照机制。
-- 主表 sys_ai_agent 字段保持为"当前可编辑态"；更新 Agent 时生成草稿快照（status=1），
--   发布动作将可编辑态序列化为 snapshot 写入本表并转为已发布（status=2）。
-- snapshot(JSON) 存完整配置快照（系统提示词、模型、推理参数、Skills/MCP/子 Agent 关联、
--   权限、护栏），运行时据此组装，与主表可编辑态解耦。
-- version_no 每个 Agent 内自增，不可变、不回填；(agent_id, version_no) 唯一。
-- 同一 Agent 同一时刻至多一条 status=2（已发布）。
-- 发布/回滚仅影响新会话，进行中会话锚定创建时的版本号（见 §4.4）。
-- 版本历史不物理删除、不使用逻辑删除（只追加）；回滚生成新版本号，完整历史可追溯、可对比。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_version`;
CREATE TABLE `sys_ai_agent_version`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `agent_id`    bigint                                                          NOT NULL COMMENT '关联Agent ID(关联sys_ai_agent.id)',
    `version_no`  int                                                             NOT NULL COMMENT '版本号(每个Agent内自增,不可变、不回填)',
    `snapshot`    json                                                            NOT NULL COMMENT '配置快照JSON(系统提示词/模型/推理参数/Skills/MCP/子Agent关联/权限/护栏的完整序列化)',
    `status`      tinyint                                                         NOT NULL DEFAULT 1 COMMENT '版本状态(1:草稿;2:已发布;同一Agent同一时刻至多一条已发布)',
    `change_note` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '变更说明',
    `operator_id` bigint                                                          NULL DEFAULT NULL COMMENT '操作人ID',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_agent_version` (`agent_id`, `version_no`) USING BTREE,
    INDEX `idx_agent` (`agent_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体配置版本快照表'
  ROW_FORMAT = DYNAMIC;
