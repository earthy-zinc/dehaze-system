-- ============================================================
-- 表名: sys_payment_record
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 支付流水表，记录每次支付请求与回调。
-- payment_no 唯一索引存储支付渠道流水号，配合 order_id 实现回调幂等（订单号+渠道流水号）。
-- channel 标识支付渠道，amount 为本笔支付金额（组合支付时一个订单多条记录）。
-- status 标识支付状态（处理中/成功/失败），callback_time 记录回调到达时间。
-- callback_content 原始保留渠道回调报文，便于对账和排查。
-- 流水记录为只追加，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_payment_record`;
CREATE TABLE `sys_payment_record`
(
    `id`               bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `order_id`         bigint                                                         NOT NULL COMMENT '订单ID',
    `user_id`          bigint                                                         NOT NULL COMMENT '用户ID',
    `payment_no`       varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付渠道流水号',
    `channel`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付渠道(wechat:微信;alipay:支付宝;balance:平台余额)',
    `amount`           bigint                                                         NOT NULL COMMENT '支付金额（单位：分）',
    `status`           tinyint                                                        NOT NULL DEFAULT 1 COMMENT '支付状态(1:处理中;2:成功;3:失败)',
    `callback_time`    datetime                                                       NULL DEFAULT NULL COMMENT '回调到达时间',
    `callback_content` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '渠道回调原始报文',
    `error_message`    varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '错误信息',
    `deleted`          tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`        bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`        bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_payment_no` (`payment_no`) USING BTREE,
    INDEX `idx_order_id` (`order_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '支付流水表'
  ROW_FORMAT = DYNAMIC;
