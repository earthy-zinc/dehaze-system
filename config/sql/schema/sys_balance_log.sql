-- ============================================================
-- 表名: sys_balance_log
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 平台余额变动流水表，记录余额账户的每一笔变动，支撑资金审计追溯与对账。
-- change_type 区分变动类型：recharge(充值)/consume(消费扣减)/refund(退款退回)/freeze(冻结)/unfreeze(解冻)。
-- amount 正数表示增加、负数表示扣减（单位：分）；balance_after 记录变动后可用余额。
-- related_id 关联业务记录（如订单ID）。
-- 只追加表，不逻辑删除（资金流水不可删），过期数据通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_balance_log`;
CREATE TABLE `sys_balance_log`
(
    `id`            bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`       bigint                                                          NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `change_type`   varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '变动类型(recharge:充值;consume:消费;refund:退款退回;freeze:冻结;unfreeze:解冻)',
    `amount`        bigint                                                          NOT NULL COMMENT '变动金额（单位：分，正数增加;负数扣减）',
    `balance_after` bigint                                                          NOT NULL COMMENT '变动后可用余额（单位：分）',
    `related_id`    bigint                                                          NULL DEFAULT NULL COMMENT '关联业务记录ID（如订单ID）',
    `deleted`       tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`   datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`     bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_create_time` (`user_id`, `create_time`) USING BTREE,
    INDEX `idx_change_type` (`change_type`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '平台余额变动流水表'
  ROW_FORMAT = DYNAMIC;
