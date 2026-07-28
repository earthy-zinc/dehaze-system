-- ============================================================
-- 表名: sys_promotion_package
-- 模块: 商业化模块-套餐管理
-- ============================================================
-- 设计思路:
-- 促销活动与套餐的多对多关联表。
-- (promotion_id, package_id) 唯一索引防止重复关联。
-- discount_type 标识折扣方式（百分比/固定金额），discount_value 存储折扣值。
-- 单独建表而非 JSON 字段，便于按套餐维度查询活动，支撑套餐列表页展示促销信息。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_promotion_package`;
CREATE TABLE `sys_promotion_package`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `promotion_id`    bigint                                                         NOT NULL COMMENT '促销活动ID',
    `package_id`      bigint                                                         NOT NULL COMMENT '套餐ID',
    `discount_type`   varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '折扣类型(percent:百分比;fixed:固定金额)',
    `discount_value` bigint                                                         NOT NULL DEFAULT 0 COMMENT '折扣值（百分比时为0-100，固定金额时为分）',
    `create_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '更新人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_promotion_package` (`promotion_id`, `package_id`) USING BTREE,
    INDEX `idx_package_id` (`package_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '促销活动-套餐关联表'
  ROW_FORMAT = DYNAMIC;
