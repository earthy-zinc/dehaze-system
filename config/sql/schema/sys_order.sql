-- ============================================================
-- 表名: sys_order
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 订单主表，记录完整交易链路。
-- order_no 唯一索引，格式为 DH + yyyyMMddHHmmss + 6位随机数，前端展示用。
-- 套餐信息冗余存储（package_name/package_level/period_days），防止套餐改名影响历史订单。
-- 金额字段统一 bigint 存分，避免浮点精度问题。
-- status 使用 6 状态机（待支付/已支付/已完成/已取消/退款中/已退款），配合定时任务驱动状态流转。
-- expire_time 为支付超时时间（创建时间+30min），定时任务每5分钟扫描自动取消。
-- coupon_id 关联 sys_user_coupon.id，下单时锁定，支付成功后核销，取消时释放。
-- is_auto_renew 标识自动续费生成的订单，与 sys_auto_renew 关联。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_order`;
CREATE TABLE `sys_order`
(
    `id`                  bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `order_no`            varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '订单号(DH+时间戳+6位随机数)',
    `user_id`             bigint                                                         NOT NULL COMMENT '用户ID',
    `package_id`          bigint                                                         NOT NULL COMMENT '套餐ID',
    `package_name`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '套餐名称（冗余）',
    `package_level`      varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '套餐对应会员等级',
    `period_days`        int                                                            NOT NULL COMMENT '有效期天数',
    `original_price`     bigint                                                         NOT NULL COMMENT '原价（单位：分）',
    `discount_amount`    bigint                                                         NOT NULL DEFAULT 0 COMMENT '促销折扣金额（单位：分）',
    `coupon_id`          bigint                                                         NULL DEFAULT NULL COMMENT '用户优惠券实例ID',
    `coupon_amount`      bigint                                                         NOT NULL DEFAULT 0 COMMENT '优惠券抵扣金额（单位：分）',
    `payable_amount`     bigint                                                         NOT NULL COMMENT '应付金额（单位：分）',
    `paid_amount`        bigint                                                         NOT NULL DEFAULT 0 COMMENT '实付金额（单位：分）',
    `pay_method`         varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '支付方式(wechat:微信;alipay:支付宝;balance:平台余额;combined:组合)',
    `status`             tinyint                                                        NOT NULL DEFAULT 1 COMMENT '订单状态(1:待支付;2:已支付;3:已完成;4:已取消;5:退款中;6:已退款)',
    `expire_time`        datetime                                                       NOT NULL COMMENT '支付超时时间（创建时间+30min）',
    `effective_time`     datetime                                                       NULL DEFAULT NULL COMMENT '权益生效时间',
    `package_expire_time` datetime                                                      NULL DEFAULT NULL COMMENT '套餐到期时间',
    `paid_time`         datetime                                                       NULL DEFAULT NULL COMMENT '支付成功时间',
    `cancel_reason`     varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '取消原因',
    `is_auto_renew`     tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否自动续费订单(0:否;1:是)',
    `deleted`           tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`         bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_order_no` (`order_no`) USING BTREE,
    INDEX `idx_user_id_status` (`user_id`, `status`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_expire_time` (`expire_time`) USING BTREE,
    INDEX `idx_package_expire_time` (`package_expire_time`) USING BTREE,
    INDEX `idx_coupon_id` (`coupon_id`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '订单表'
  ROW_FORMAT = DYNAMIC;
