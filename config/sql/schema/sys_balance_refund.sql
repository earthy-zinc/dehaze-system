-- ============================================================
-- 表名: sys_balance_refund
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 平台余额退款（充值余额退回）记录表，与订单售后退款（sys_refund_record）隔离。
-- 余额退款仅针对充值余额本身，金额 = 用户可用余额（不含冻结），不触发任何订单履约回退。
-- status: 1 待审核 / 2 已退款 / 3 退款失败。
-- channel 记录原路退回渠道（微信/支付宝，无法原路由管理员指定）。
-- 标准逻辑删除（类别②，user_id 为业务引用键，删除后不可复用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_balance_refund`;
CREATE TABLE `sys_balance_refund`
(
    `id`                 bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `refund_no`          varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '退款单号',
    `user_id`            bigint                                                         NOT NULL COMMENT '用户ID',
    `amount`             bigint                                                         NOT NULL COMMENT '退款金额（单位：分，=申请时可用余额）',
    `status`             tinyint                                                        NOT NULL DEFAULT 1 COMMENT '退款状态(1:待审核;2:已退款;3:退款失败)',
    `channel`            varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '原路退回渠道(wechat:微信;alipay:支付宝)',
    `channel_refund_no`  varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '渠道退款流水号',
    `apply_time`         datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '申请时间',
    `audit_time`         datetime                                                       NULL DEFAULT NULL COMMENT '审核时间',
    `auditor_id`         bigint                                                         NULL DEFAULT NULL COMMENT '审核人ID',
    `audit_remark`       varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '审核备注',
    `refund_time`        datetime                                                       NULL DEFAULT NULL COMMENT '退款完成时间',
    `error_message`      varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '错误信息',
    `deleted`            tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`        datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`        datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`          bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`          bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_refund_no` (`refund_no`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '平台余额退款记录表'
  ROW_FORMAT = DYNAMIC;
