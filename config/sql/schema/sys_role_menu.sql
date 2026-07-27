-- ============================================================
-- 表名: sys_role_menu
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 角色-菜单多对多关联表。通过 role_id 关联角色，menu_id 关联菜单。
-- 关联表不使用逻辑删除与审计字段，角色权限变更直接覆盖。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_role_menu`;
CREATE TABLE `sys_role_menu`
(
    `role_id` bigint NOT NULL COMMENT '角色ID',
    `menu_id` bigint NOT NULL COMMENT '菜单ID',
    PRIMARY KEY (`role_id`, `menu_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '角色和菜单关联表'
  ROW_FORMAT = DYNAMIC;
