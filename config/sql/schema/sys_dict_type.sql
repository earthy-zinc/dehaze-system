-- ============================================================
-- 表名: sys_dict_type
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 字典类型表，管理字典分类。code 字段唯一索引，作为业务层引用键。
-- 字典类型和字典数据拆分为两张表，避免冗余存储类型信息。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_dict_type`;
CREATE TABLE `sys_dict_type`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '类型名称',
    `code`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '类型编码',
    `status`      tinyint                                                       NULL DEFAULT 1 COMMENT '状态(1:正常;0:禁用)',
    `remark`      varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '备注',
    `deleted`     tinyint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '字典类型表'
  ROW_FORMAT = DYNAMIC;
