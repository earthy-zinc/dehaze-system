-- ============================================================
-- 表名: sys_auto_renew
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 自动续费配置表，每个用户对每个套餐一条配置。
-- (user_id, package_id) 唯一索引保证配置唯一。
-- status 标识启用/关闭，关闭时记录 close_reason。
-- next_renew_time 为下次扣款时间，定时任务每小时扫描此字段到期记录。
-- fail_count 记录连续失败次数，达到3次后自动关闭续费。
-- last_renew_order_id 关联上次续费订单，便于追溯扣款历史。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_auto_renew`;
CREATE TABLE `sys_auto_renew`
(
    `id`                  bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`             bigint                                                         NOT NULL COMMENT '用户ID',
    `package_id`          bigint                                                         NOT NULL COMMENT '套餐ID',
    `pay_method`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付方式(wechat:微信;alipay:支付宝;balance:平台余额)',
    `status`              tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:已关闭)',
    `next_renew_time`     datetime                                                       NULL DEFAULT NULL COMMENT '下次扣款时间',
    `fail_count`          int                                                            NOT NULL DEFAULT 0 COMMENT '连续失败次数',
    `last_renew_order_id` bigint                                                         NULL DEFAULT NULL COMMENT '上次续费订单ID',
    `close_reason`        varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '关闭原因',
    `deleted`             tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`           bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`           bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_package` (`user_id`, `package_id`) USING BTREE,
    INDEX `idx_status_renew_time` (`status`, `next_renew_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '自动续费配置表'
  ROW_FORMAT = DYNAMIC;
