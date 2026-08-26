-- ============================================================
-- 表名: sys_balance
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 平台人民币余额账户表（交易媒介），支撑余额支付与组合支付（F-OM-015）。
-- user_id 唯一，每用户一行。
-- balance/frozen_balance 单位统一为分（bigint），避免浮点精度问题。
-- version 乐观锁版本号，扣减使用条件 UPDATE 防并发超卖。
-- 与 AI 积分账户（sys_ai_credit_log）职责分离：本账户为人民币交易媒介，
--   积分（AI 消耗计量）由 AI 计费模块积分账户管理。
-- 标准逻辑删除（类别②，user_id 为业务引用键，删除后不可复用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_balance`;
CREATE TABLE `sys_balance`
(
    `id`             bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`        bigint                                                         NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `balance`        bigint                                                         NOT NULL DEFAULT 0 COMMENT '可用余额（单位：分）',
    `frozen_balance` bigint                                                         NOT NULL DEFAULT 0 COMMENT '冻结余额（单位：分，支付处理中冻结）',
    `version`        int                                                            NOT NULL DEFAULT 0 COMMENT '乐观锁版本号',
    `deleted`        tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_id` (`user_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '平台余额账户表'
  ROW_FORMAT = DYNAMIC;
