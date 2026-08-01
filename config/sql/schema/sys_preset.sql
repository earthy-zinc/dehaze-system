-- ============================================================
-- 表名: sys_preset
-- 模块: 核心模块-去雾处理-参数预设
-- ============================================================
-- 设计思路:
-- 参数预设表，支持系统预设(type=system)和用户自定义预设(type=custom)。
-- 系统预设对所有用户可见且只读，用户自定义预设仅本人可操作。
-- algorithm_id + params 存储预设关联的算法和参数配置。
-- uk_user_name 确保同一用户下预设名称唯一（系统预设名称全局唯一）。
-- is_default 标识是否为默认预设。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_preset`;
CREATE TABLE `sys_preset`
(
    `id`           bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '预设名称',
    `type`         varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'custom' COMMENT '预设类型(system:系统预设;custom:用户自定义)',
    `algorithm_id` bigint                                                        NOT NULL COMMENT '关联算法ID',
    `params`       json                                                          NULL COMMENT '参数键值对(JSON)',
    `user_id`      bigint                                                        NULL DEFAULT NULL COMMENT '所属用户ID(系统预设为空)',
    `is_default`   tinyint                                                       NOT NULL DEFAULT 0 COMMENT '是否默认预设(0:否;1:是)',
    `create_time`  datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    UNIQUE INDEX `uk_user_name` (`user_id`, `name`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '参数预设表'
  ROW_FORMAT = DYNAMIC;