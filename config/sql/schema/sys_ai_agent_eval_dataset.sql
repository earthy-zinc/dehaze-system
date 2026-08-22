-- ============================================================
-- 表名: sys_ai_agent_eval_dataset
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 智能体评测集表，挂载在 Agent 下，按 dataset_type 分层管理：
--   dev       开发集：日常调试
--   regression 回归集：发布前必跑门禁
--   heldout   保留集：阶段验收
-- 评测集挂载在 Agent 下，样本（sys_ai_agent_eval_sample）属于评测集。
-- 配置类表，使用逻辑删除；唯一键 (agent_id, dataset_type) 不含 deleted（类别①，upsert 复活）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_eval_dataset`;
CREATE TABLE `sys_ai_agent_eval_dataset`
(
    `id`           bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `agent_id`     bigint                                                          NOT NULL COMMENT '关联Agent ID(关联sys_ai_agent.id)',
    `name`         varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '评测集名称',
    `description`  varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT '' COMMENT '评测集描述',
    `dataset_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '评测集类型(dev:开发集;regression:回归集;heldout:保留集)',
    `deleted`      tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`    bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`  datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_agent_dataset_type` (`agent_id`, `dataset_type`) USING BTREE,
    INDEX `idx_agent` (`agent_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体评测集表'
  ROW_FORMAT = DYNAMIC;
