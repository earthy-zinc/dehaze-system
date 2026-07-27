-- ============================================================
-- 表名: sys_dict
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 字典数据表，存储各模块的枚举值（性别、算法类型等）。
-- 通过 type_code 关联 sys_dict_type，运行时缓存到 Redis 减少查库。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `type_code`   varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '字典类型编码',
    `name`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '字典项名称',
    `value`       varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '字典项值',
    `sort`        int                                                           NULL DEFAULT 0 COMMENT '排序',
    `status`      tinyint                                                       NULL DEFAULT 1 COMMENT '状态(1:正常;0:禁用)',
    `defaulted`   tinyint                                                       NULL DEFAULT 0 COMMENT '是否默认(1:是;0:否)',
    `remark`      varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '备注',
    `deleted`     tinyint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '字典数据表'
  ROW_FORMAT = DYNAMIC;
