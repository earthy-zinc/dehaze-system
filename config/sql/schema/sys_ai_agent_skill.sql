-- ============================================================
-- 表名: sys_ai_agent_skill
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- Agent-Skill 多对多关联表。通过 agent_id 关联智能体，skill_name 关联 Skill 名称。
-- skill_name 关联 sys_ai_skill.name（Skills 管理在 F-M08-006 能力扩展体系，Skill 的启停
--   由 sys_ai_skill.status 控制，本表仅记录 Agent 与 Skill 的关联关系）。
-- DehazeToolsBuilder 按 Agent 配置从此表查询关联的 Skills，通过 skill_load 工具渐进式加载。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_skill`;
CREATE TABLE `sys_ai_agent_skill`
(
    `agent_id`    bigint                                                         NOT NULL COMMENT '关联Agent ID(关联sys_ai_agent.id)',
    `skill_name`  varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT 'Skill名称(关联sys_ai_skill.name)',
    `create_time` datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`agent_id`, `skill_name`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'Agent-Skill关联表'
  ROW_FORMAT = DYNAMIC;
