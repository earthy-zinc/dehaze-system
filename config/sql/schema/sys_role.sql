-- ============================================================
-- 表名: sys_role
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 角色表，data_scope 字段控制数据权限范围（0全部/1部门及子部门/2本部门/3本人）。
-- 角色名称唯一，防止重复创建。逻辑删除保留历史关联记录。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_role`;
CREATE TABLE `sys_role`
(
    `id`          bigint                                                       NOT NULL AUTO_INCREMENT,
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL DEFAULT '' COMMENT '角色名称',
    `code`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL     DEFAULT NULL COMMENT '角色编码',
    `sort`        int                                                          NULL     DEFAULT NULL COMMENT '显示顺序',
    `status`      tinyint(1)                                                   NULL     DEFAULT 1 COMMENT '角色状态(1-正常；0-停用)',
    `data_scope`  tinyint                                                      NULL     DEFAULT NULL COMMENT '数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)',
    `deleted`     tinyint(1)                                                   NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0-未删除；1-已删除)',
    `create_time` datetime                                                     NULL     DEFAULT NULL COMMENT '更新时间',
    `update_time` datetime                                                     NULL     DEFAULT NULL COMMENT '创建时间',
    `create_by`   bigint                                                       NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                       NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `name` (`name` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '角色表'
  ROW_FORMAT = DYNAMIC;
