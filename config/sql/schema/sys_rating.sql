-- ============================================================
-- 表名: sys_rating
-- 模块: 商业化模块-反馈评价
-- ============================================================
-- 设计思路:
-- 处理结果评分评价表，记录用户对去雾结果的评价。
-- pred_log_id 唯一索引保证每条处理记录仅可评价一次（防刷）。
-- rating 字段为 1-5 星整数，comment 最多 500 字符。
-- tags 使用 JSON 数组存储预设标签（去雾彻底/残留雾气等），便于聚合统计。
-- image_urls 使用 JSON 数组存储截图URL（最多3张），文件存储走 MinIO。
-- is_anonymous 控制匿名评价，用户端不展示用户信息但后台可见。
-- is_hidden 由后台隐藏不当评价，用户端不展示但保留数据。
-- admin_reply/admin_reply_time 记录管理员回复。
-- 低分评价（rating<=2）由后端逻辑触发告警通知，不单独建告警表。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_rating`;
CREATE TABLE `sys_rating`
(
    `id`             bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`        bigint                                                         NOT NULL COMMENT '评价用户ID',
    `pred_log_id`    bigint                                                         NOT NULL COMMENT '关联处理日志ID（sys_pred_log.id）',
    `algorithm_id`   bigint                                                         NOT NULL COMMENT '算法ID',
    `rating`         tinyint                                                        NOT NULL COMMENT '评分(1-5星)',
    `comment`        varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '评价文字',
    `tags`           json                                                           NULL DEFAULT NULL COMMENT '评价标签（JSON数组）',
    `image_urls`     json                                                           NULL DEFAULT NULL COMMENT '截图URL（JSON数组，最多3张）',
    `is_anonymous`   tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否匿名(0:否;1:是)',
    `is_hidden`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否隐藏(0:否;1:是)',
    `admin_reply`    varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '管理员回复内容',
    `reply_time`     datetime                                                       NULL DEFAULT NULL COMMENT '管理员回复时间',
    `deleted`        tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_pred_log_id` (`pred_log_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    INDEX `idx_rating` (`rating`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '处理结果评分评价表'
  ROW_FORMAT = DYNAMIC;
