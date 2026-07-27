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
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `dataset_id`  bigint      NOT NULL COMMENT '所属数据集id',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '数据项名称',
    `deleted`     tinyint     NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime    NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime    NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint      NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint      NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_dataset_id` (`dataset_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '数据集与数据项关联表'
  ROW_FORMAT = DYNAMIC;
