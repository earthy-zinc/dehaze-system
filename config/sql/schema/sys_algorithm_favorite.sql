-- ============================================================
-- 表名: sys_algorithm_favorite
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 算法收藏表，uk_user_algorithm 唯一索引防止重复收藏。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_algorithm_favorite`;
CREATE TABLE `sys_algorithm_favorite`
(
    `id`            bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `user_id`       bigint      NOT NULL COMMENT '用户ID',
    `algorithm_id`  bigint      NOT NULL COMMENT '算法ID',
    `create_time`   datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_algorithm` (`user_id`, `algorithm_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法收藏表';
