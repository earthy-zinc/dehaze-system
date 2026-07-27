-- ============================================================
-- 表名: sys_user_role
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 用户-角色多对多关联表。联合主键 (user_id, role_id) 防止重复分配。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_user_role`;
CREATE TABLE `sys_user_role`
(
    `user_id` bigint NOT NULL COMMENT '用户ID',
    `role_id` bigint NOT NULL COMMENT '角色ID',
    PRIMARY KEY (`user_id`, `role_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '用户和角色关联表'
  ROW_FORMAT = DYNAMIC;
