-- ============================================================
-- 表名: sys_package
-- 模块: 商业化模块-套餐管理
-- ============================================================
-- 设计思路:
-- 套餐表，定义商品定价与计费周期。
-- name 唯一索引保证套餐名称全局唯一。
-- level_code 关联会员等级，period 标识计费周期（月/季/年）。
-- 价格字段统一使用 bigint 存储分，避免浮点精度问题。
-- sale_price 为促销价，original_price 为原价，前端展示删除线效果。
-- benefit_overrides 使用 JSON 存储套餐对等级权益的覆盖项，避免单独建表。
-- status 控制上下架（1:上架;0:下架），下架后用户端不可见。
-- sales_count 销量由订单完成时异步累加，支撑运营排序。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_package`;
CREATE TABLE `sys_package`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`              varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '套餐名称',
    `level_code`        varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '关联会员等级(level_1/level_2/level_3)',
    `period`            varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '计费周期(monthly:月;quarterly:季;yearly:年)',
    `period_days`       int                                                            NOT NULL COMMENT '有效期天数',
    `original_price`    bigint                                                         NOT NULL COMMENT '原价（单位：分）',
    `sale_price`        bigint                                                         NOT NULL COMMENT '促销价（单位：分）',
    `description`       varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '套餐描述',
    `benefit_overrides` json                                                           NULL DEFAULT NULL COMMENT '套餐权益覆盖项（JSON，覆盖等级默认权益）',
    `sales_count`       bigint                                                         NOT NULL DEFAULT 0 COMMENT '销量',
    `sort`              int                                                            NOT NULL DEFAULT 0 COMMENT '排序值',
    `status`            tinyint                                                        NOT NULL DEFAULT 0 COMMENT '上下架状态(1:上架;0:下架)',
    `deleted`           tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`         bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_name` (`name`) USING BTREE,
    INDEX `idx_level_code` (`level_code`) USING BTREE,
    INDEX `idx_period` (`period`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '套餐表'
  ROW_FORMAT = DYNAMIC;
