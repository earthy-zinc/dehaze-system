-- ============================================================
-- 表名: sys_dataset_item
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 数据集项表，每个数据项属于一个数据集，通过 dataset_id 关联。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_dataset_item`;
CREATE TABLE `sys_dataset_item`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `dataset_id`  bigint      NOT NULL COMMENT '所属数据集id',
    `name`        varchar(64) NULL COMMENT '数据项名称',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_dataset_id` (`dataset_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据集与数据项关联表';
