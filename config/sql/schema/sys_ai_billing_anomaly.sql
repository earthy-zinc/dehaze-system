-- ============================================================
-- 表名: sys_ai_billing_anomaly
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- AI 计费异常事件记录表，承载异常检测四类规则的命中结果：
-- single_high(单次超高消耗)/burst(突发峰值)/consecutive_quota_fail(连续配额不足)/empty_high_output(空回复高消耗)。
-- 记录与 Redis 告警计数（ai:anomaly:count:{type}:{user_id}）并存：Redis 负责实时告警，
-- 本表负责审计与异常清单查询（管理员按类型/时间筛选），Redis 不可用时告警计数降级、落库不受影响。
-- billing_id 关联原计费记录（sys_ai_billing.id）；配额不足类异常在预扣前触发，无关联记录时为 NULL。
-- 只追加表，不逻辑删除（异常事件为审计数据），过期数据通过定时任务物理清理；
-- 日志表不承载通用操作人审计（create_by/update_by），仅保留 create_time。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_billing_anomaly`;
CREATE TABLE `sys_ai_billing_anomaly`
(
    `id`           bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`      bigint                                                          NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `billing_id`   bigint                                                          NULL DEFAULT NULL COMMENT '计费记录ID(关联sys_ai_billing.id,配额不足类异常无关联记录)',
    `anomaly_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '异常类型(single_high;burst;consecutive_quota_fail;empty_high_output)',
    `detail`       varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '异常详情',
    `status`       tinyint                                                         NOT NULL DEFAULT 0 COMMENT '处理状态(0:待处理;1:已处理;2:已忽略)',
    `trigger_at`   datetime                                                        NOT NULL COMMENT '触发时间',
    `create_time`  datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_type` (`user_id`, `anomaly_type`) USING BTREE,
    INDEX `idx_type_trigger` (`anomaly_type`, `trigger_at`) USING BTREE,
    INDEX `idx_billing_id` (`billing_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI计费异常事件记录表'
  ROW_FORMAT = DYNAMIC;
