-- ============================================================
-- 表名: sys_user_coupon
-- 模块: 商业化模块-套餐管理
-- ============================================================
-- 设计思路:
-- 用户优惠券实例表，用户领取优惠券后生成一条记录。
-- status 标识券状态（未使用/已使用/已过期/已锁定），锁定状态用于下单时预占优惠券。
-- expire_time 为实例过期时间，由 fixed 类型券按 coupon.valid_end 计算，relative 类型按 receive_time + valid_days 计算。
-- used_order_id 记录使用的订单，便于退款时追溯（退款不退券）。
-- (user_id, coupon_id) 不设唯一索引，因为 per_user_limit 可能允许多次领取。
-- 定时任务扫描 expire_time < NOW() 且 status=未使用 的记录，更新为已过期。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_user_coupon`;
CREATE TABLE `sys_user_coupon`
(
    `id`            bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`       bigint                                                         NOT NULL COMMENT '用户ID',
    `coupon_id`    bigint                                                         NOT NULL COMMENT '优惠券模板ID',
    `status`        tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:未使用;2:已使用;3:已过期;4:已锁定)',
    `receive_time` datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '领取时间',
    `expire_time`  datetime                                                       NULL DEFAULT NULL COMMENT '过期时间',
    `used_time`    datetime                                                       NULL DEFAULT NULL COMMENT '使用时间',
    `used_order_id` bigint                                                        NULL DEFAULT NULL COMMENT '使用的订单ID',
    `deleted`       tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id_status` (`user_id`, `status`) USING BTREE,
    INDEX `idx_coupon_id` (`coupon_id`) USING BTREE,
    INDEX `idx_expire_time` (`expire_time`) USING BTREE,
    INDEX `idx_used_order_id` (`used_order_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户优惠券实例表'
  ROW_FORMAT = DYNAMIC;
