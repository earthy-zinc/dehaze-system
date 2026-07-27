-- ============================================================
-- 表名: sys_coupon
-- 模块: 商业化模块-套餐管理
-- ============================================================
-- 设计思路:
-- 优惠券模板表，定义优惠券规则与发放库存。
-- type 标识四种类型：满减券/折扣券/无门槛券/体验券。
-- face_value/threshold 配合 type 使用：满减券需 threshold+face_value，折扣券 face_value 为折扣比例（0-100）。
-- valid_type 区分固定时间段和领取后N天两种有效期模式，对应字段分别填充。
-- total_qty 为发放总量（-1表示不限），issued_qty/used_qty 由领取和核销时累加，配合乐观锁防止超发。
-- applicable_scope 为 JSON 数组，存储适用套餐ID（NULL表示全部适用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_coupon`;
CREATE TABLE `sys_coupon`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '优惠券名称',
    `type`              varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '类型(full_reduction:满减券;discount:折扣券;no_threshold:无门槛券;trial:体验券)',
    `face_value`        bigint                                                         NOT NULL DEFAULT 0 COMMENT '面值（满减/无门槛为分，折扣为0-100）',
    `threshold`         bigint                                                         NULL DEFAULT NULL COMMENT '使用门槛（满减券必填，单位：分）',
    `valid_type`        varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '有效期类型(fixed:固定时间段;relative:领取后N天)',
    `valid_start`       datetime                                                       NULL DEFAULT NULL COMMENT '有效期开始时间（fixed类型必填）',
    `valid_end`         datetime                                                       NULL DEFAULT NULL COMMENT '有效期结束时间（fixed类型必填）',
    `valid_days`        int                                                            NULL DEFAULT NULL COMMENT '领取后有效天数（relative类型必填）',
    `total_qty`         int                                                            NOT NULL DEFAULT -1 COMMENT '发放总量（-1为不限）',
    `issued_qty`        int                                                            NOT NULL DEFAULT 0 COMMENT '已发放数量',
    `used_qty`          int                                                            NOT NULL DEFAULT 0 COMMENT '已使用数量',
    `per_user_limit`    int                                                            NOT NULL DEFAULT 1 COMMENT '每人限领数量',
    `applicable_scope`  json                                                           NULL DEFAULT NULL COMMENT '适用套餐ID列表（JSON数组，NULL表示全部）',
    `status`            tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`           tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`         bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_type` (`type`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '优惠券模板表'
  ROW_FORMAT = DYNAMIC;
