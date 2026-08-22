-- ============================================================
-- 表名: sys_ai_agent_eval_run
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 智能体评测执行记录表。手动触发（manual）或发布门禁内部触发（publish）。
-- score_summary(JSON) 存四维评分聚合（结果质量/过程合规/安全边界/效率）；
-- results(JSON) 存每条样本明细（四维分数 + 通过状态 + 差异说明），不独立建结果表。
-- 发布审计轨迹：发布成功后 eval_run 保留，支持回溯每次发布对应的评测结果（见 §9.3）。
-- 日志/历史类表，只追加记录，不删除、不使用逻辑删除；过期数据由定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent_eval_run`;
CREATE TABLE `sys_ai_agent_eval_run`
(
    `id`            bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `agent_id`      bigint                                                         NOT NULL COMMENT '关联Agent ID(关联sys_ai_agent.id)',
    `dataset_id`    bigint                                                         NOT NULL COMMENT '关联评测集ID(关联sys_ai_agent_eval_dataset.id)',
    `trigger_type`  varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '触发方式(manual:手动触发;publish:发布门禁内部触发)',
    `status`        tinyint                                                        NOT NULL DEFAULT 1 COMMENT '执行状态(1:执行中;2:通过;3:失败)',
    `score_summary` json                                                           NULL COMMENT '四维评分聚合JSON(结果质量/过程合规/安全边界/效率)',
    `results`       json                                                           NULL COMMENT '样本明细JSON(每条样本的四维分数+通过状态+差异说明)',
    `create_by`     bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID(触发评测的用户)',
    `create_time`   datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_agent` (`agent_id`) USING BTREE,
    INDEX `idx_dataset` (`dataset_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体评测执行记录表'
  ROW_FORMAT = DYNAMIC;
