-- ============================================================
-- 表名: sys_member_sign_in
-- 模块: 商业化模块-会员管理
-- ============================================================
-- 设计思路:
-- 签到记录表，每日一条记录。
-- (user_id, sign_date) 唯一索引防止重复签到。
-- continuous_days 记录签到时的连续天数，便于计算7天额外奖励。
-- growth_value 记录本次签到获得的成长值（含连续签到奖励）。
-- 签到记录为只追加，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_member_sign_in`;
CREATE TABLE `sys_member_sign_in`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`         bigint                                                         NOT NULL COMMENT '用户ID',
    `sign_date`       date                                                           NOT NULL COMMENT '签到日期',
    `continuous_days` int                                                            NOT NULL DEFAULT 1 COMMENT '连续签到天数',
    `growth_value`    int                                                            NOT NULL DEFAULT 0 COMMENT '本次获得成长值',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_sign_date` (`user_id`, `sign_date`) USING BTREE,
    INDEX `idx_sign_date` (`sign_date`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '会员签到记录表'
  ROW_FORMAT = DYNAMIC;
