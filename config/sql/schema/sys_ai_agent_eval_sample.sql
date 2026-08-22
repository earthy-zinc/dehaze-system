-- ============================================================
-- 表名: sys_ai_agent_eval_sample
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 智能体评测样本表，属于某个评测集（sys_ai_agent_eval_dataset）。
-- 每条样本含可审计的通过/失败条件：任务目标、允许输入、工具、期望过程、期望结果、禁止行为。
-- risk_level 标记风险等级，high 样本在发布门禁中失败即硬阻断发布（见 §9.3）。
-- Bad Case 脱敏回流复用"创建样本"接口写入开发集/回归集，防止同类问题复现。
-- 随所属评测集管理（评测集软删时样本一并失效），不单独使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_eval_sample`;
CREATE TABLE `sys_ai_agent_eval_sample`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `dataset_id`        bigint                                                         NOT NULL COMMENT '关联评测集ID(关联sys_ai_agent_eval_dataset.id)',
    `task_goal`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NOT NULL COMMENT '任务目标(样本要完成的任务描述)',
    `allowed_input`     TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '允许输入(输入范围/约束说明,可空)',
    `tools`             json                                                           NULL COMMENT '可用工具JSON(样本预期调用的工具列表,可空)',
    `expected_process`  TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '期望过程(正确的推理/调用过程,可空)',
    `expected_result`   TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '期望结果(准确/完整/格式等通过条件,可空)',
    `forbidden_behavior` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '禁止行为(不得发生的越权/注入/敏感泄露等,可空)',
    `risk_level`        varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'low' COMMENT '风险等级(low:低;medium:中;high:高,high样本失败阻断发布)',
    `create_by`         bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`       datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_dataset` (`dataset_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体评测样本表'
  ROW_FORMAT = DYNAMIC;
