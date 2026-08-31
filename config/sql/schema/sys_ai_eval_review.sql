-- ============================================================
-- 表名: sys_ai_eval_review
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 智能体评测人工复核表。评测完成后，失败样本全量 + 通过样本按比例
-- （sys_dict ai_eval.judge_review_ratio）抽样生成待复核项（uk_run_sample 幂等去重），
-- 人工复核结果（判定一致/不一致）回填用于判分漂移检测与判分校准。
-- 复核记录只追加一次判定，重复复核不允许（保护回填数据可信度）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_eval_review`;
CREATE TABLE `sys_ai_eval_review`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `run_id`       bigint                                                         NOT NULL COMMENT '关联评测执行ID(关联sys_ai_agent_eval_run.id)',
    `sample_id`    bigint                                                         NOT NULL COMMENT '关联评测样本ID(关联sys_ai_agent_eval_sample.id)',
    `agent_id`     bigint                                                         NOT NULL COMMENT '关联Agent ID(冗余自sys_ai_agent_eval_run.agent_id,按Agent聚合查询)',
    `judge_passed` tinyint                                                        NOT NULL COMMENT '判分模型判定(1:通过;0:失败)',
    `risk_level`   varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'low' COMMENT '样本风险等级快照(low/medium/high)',
    `status`       tinyint                                                        NOT NULL DEFAULT 1 COMMENT '复核状态(1:待复核;2:已复核)',
    `agree`        tinyint                                                        NULL DEFAULT NULL COMMENT '人工判定(1:与判分一致;0:不一致)',
    `reviewer_id`  bigint                                                         NULL DEFAULT NULL COMMENT '复核人ID(关联sys_user.id)',
    `remark`       varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '复核备注',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `create_time`  datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '更新人ID',
    `update_time`  datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_run_sample` (`run_id`, `sample_id`) USING BTREE,
    INDEX `idx_agent_status` (`agent_id`, `status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体评测人工复核表'
  ROW_FORMAT = DYNAMIC;
