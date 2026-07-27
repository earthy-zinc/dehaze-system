-- ============================================================
-- 表名: sys_dept
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 采用 parent_id + tree_path 双字段实现树形结构：parent_id 记录直接父级，
-- tree_path 存储完整路径（逗号分隔），用于递归查询子树和祖先链。
-- tree_path 避免了递归 CTE 查询，通过 LIKE 'path%' 即可获取所有子部门。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_dept`;
CREATE TABLE `sys_dept`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '部门名称',
    `parent_id`   bigint                                                        NOT NULL DEFAULT 0 COMMENT '父节点id',
    `tree_path`   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL     DEFAULT '' COMMENT '父节点id路径',
    `sort`        int                                                           NULL     DEFAULT 0 COMMENT '显示顺序',
    `status`      tinyint                                                       NOT NULL DEFAULT 1 COMMENT '状态(1:正常;0:禁用)',
    `deleted`     tinyint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                      NULL     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                      NULL     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '部门表'
  ROW_FORMAT = DYNAMIC;
