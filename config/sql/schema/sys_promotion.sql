-- ============================================================
-- 表名: sys_promotion
-- 模块: 商业化模块-套餐管理
-- ============================================================
-- 设计思路:
-- 促销活动表，支持限时折扣、新用户专享、节日促销、满减活动四种类型。
-- start_time/end_time 控制活动有效期，定时任务到点自动开始/结束。
-- activity_rules 使用 JSON 存储活动规则（如满减门槛、折扣比例），避免硬编码字段。
-- 活动与商品的关联及折扣参数统一由 sys_promotion_package 关联表承载（价格计算遍历该表取最大折扣），
--   sys_promotion 不再冗余存适用范围（避免双源不一致）。
-- new_user_only 标识新用户专享活动，配合 sys_member.become_member_time 判断。
-- status 字段控制启用/禁用，与时间范围共同决定活动是否生效。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_promotion`;
CREATE TABLE `sys_promotion`
(
    `id`                bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '活动名称',
    `type`              varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '活动类型(discount:限时折扣;new_user:新用户专享;holiday:节日促销;full_reduction:满减活动)',
    `description`       varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '活动描述',
    `start_time`        datetime                                                       NOT NULL COMMENT '活动开始时间',
    `end_time`          datetime                                                       NOT NULL COMMENT '活动结束时间',
    `activity_rules`    json                                                           NULL DEFAULT NULL COMMENT '活动规则（JSON，如满减门槛、折扣比例）',
    `new_user_only`     tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否新用户专享(0:否;1:是)',
    `status`            tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`           tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`         bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_type` (`type`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_time_range` (`start_time`, `end_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '促销活动表'
  ROW_FORMAT = DYNAMIC;
