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
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`      bigint      NOT NULL COMMENT '用户ID',
    `algorithm_id` bigint      NOT NULL COMMENT '算法ID',
    `deleted`      tinyint     NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
    `update_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint      NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint      NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_algorithm` (`user_id`, `algorithm_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '算法收藏表'
  ROW_FORMAT = DYNAMIC;
