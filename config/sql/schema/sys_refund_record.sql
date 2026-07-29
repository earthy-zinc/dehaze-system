-- ============================================================
-- 表名: sys_refund_record
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 退款记录表，记录退款申请、审核、执行全流程。
-- refund_no 唯一索引为本系统退款单号，channel_refund_no 为渠道退款流水号。
-- 每个订单仅可退款一次，order_id 唯一索引强制约束。
-- status 标识退款状态（退款中/成功/失败），配合 auditor_id 和 audit_remark 记录审核信息。
-- refund_amount 为实际退款金额，由后端按使用情况计算（未使用全额/部分使用按比例）。
-- used_quota 记录申请时的权益使用情况，便于审核判断。
-- 失败时 error_message 记录渠道返回信息，支持人工重试。
-- retry_count 记录自动重试次数，达上限后不再重试。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_refund_record`;
CREATE TABLE `sys_refund_record`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `refund_no`         varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '退款单号',
    `order_id`         bigint                                                         NOT NULL COMMENT '订单ID',
    `user_id`          bigint                                                         NOT NULL COMMENT '用户ID',
    `refund_amount`    bigint                                                         NOT NULL COMMENT '退款金额（单位：分）',
    `reason`           varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '退款原因',
    `used_quota`       int                                                            NOT NULL DEFAULT 0 COMMENT '申请时已用权益次数（去雾+评估）',
    `status`           tinyint                                                        NOT NULL DEFAULT 1 COMMENT '退款状态(1:退款中;2:退款成功;3:退款失败)',
    `channel`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '退款渠道(wechat:微信;alipay:支付宝;balance:平台余额)',
    `channel_refund_no` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '渠道退款流水号',
    `apply_time`       datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '申请时间',
    `audit_time`       datetime                                                       NULL DEFAULT NULL COMMENT '审核时间',
    `auditor_id`       bigint                                                         NULL DEFAULT NULL COMMENT '审核人ID',
    `audit_remark`     varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '审核备注',
    `refund_time`      datetime                                                       NULL DEFAULT NULL COMMENT '退款完成时间',
    `error_message`    varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '错误信息',
    `retry_count`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '自动重试次数',
    `deleted`          tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`        bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`        bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_refund_no` (`refund_no`) USING BTREE,
    UNIQUE INDEX `uk_order_id` (`order_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_auditor_id` (`auditor_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '退款记录表'
  ROW_FORMAT = DYNAMIC;
