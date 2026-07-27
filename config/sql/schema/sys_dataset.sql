-- ============================================================
-- 表名: sys_dataset
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 数据集表，树形嵌套结构，parent_id 实现父子数据集层级。
-- path 字段存储磁盘路径，配合 nginx-dataset 静态服务访问数据集文件。
-- usage_count 统计被引用次数，支撑数据集热度排序。
-- 复合索引 idx_parent_name 优化「同层级下按名称查询」的高频场景。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_dataset`;
CREATE TABLE `sys_dataset`
(
    `id`          bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '数据集ID',
    `parent_id`   bigint                                                         NOT NULL DEFAULT 0 COMMENT '父数据集ID',
    `type`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT '' COMMENT '数据集类型',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT '' COMMENT '数据集名称',
    `img`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci          NULL     DEFAULT NULL COMMENT '数据集样例图片',
    `description` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL     DEFAULT '' COMMENT '数据集描述',
    `path`        varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT '' COMMENT '存储位置',
    `size`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL     DEFAULT '' COMMENT '占用空间大小',
    `status`      tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `usage_count` bigint                                                         NOT NULL DEFAULT 0 COMMENT '使用次数',
    `deleted`     tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                       NULL     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                       NULL     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_parent_id` (`parent_id`) USING BTREE,
    INDEX `idx_name` (`name`) USING BTREE,
    INDEX `idx_parent_name` (`parent_id`, `name`) USING BTREE,
    INDEX `idx_deleted` (`deleted`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '数据集表'
  ROW_FORMAT = DYNAMIC;
