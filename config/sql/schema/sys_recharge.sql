-- ============================================================
-- 表名: sys_recharge
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 平台余额充值订单表（人民币余额账户充值，需求规格 §3.5.3）。
-- 与积分卡/会员卡交易（sys_order）隔离：充值仅支持微信/支付宝渠道下单，
-- 回调成功后可用余额入账并写 sys_balance_log（change_type=recharge）。
-- status: 1 待支付 / 2 已支付 / 3 已关闭。
-- channel_payment_no 唯一约束，作为回调幂等依据之一（另一重为状态判断+分布式锁）。
-- 标准逻辑删除（类别②，user_id 为业务引用键，删除后不可复用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_recharge`;
CREATE TABLE `sys_recharge`
(
    `id`                  bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `recharge_no`         varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '充值单号',
    `user_id`             bigint                                                         NOT NULL COMMENT '用户ID',
    `amount`              bigint                                                         NOT NULL COMMENT '充值金额（单位：分）',
    `pay_method`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付方式(wechat:微信;alipay:支付宝)',
    `status`              tinyint                                                        NOT NULL DEFAULT 1 COMMENT '充值状态(1:待支付;2:已支付;3:已关闭)',
    `channel_payment_no`  varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '渠道支付流水号(回调幂等依据)',
    `pay_time`            datetime                                                       NULL DEFAULT NULL COMMENT '支付成功时间',
    `deleted`             tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`           bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`           bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_recharge_no` (`recharge_no`) USING BTREE,
    UNIQUE INDEX `uk_channel_payment_no` (`channel_payment_no`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '余额充值订单表'
  ROW_FORMAT = DYNAMIC;
