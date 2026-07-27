-- ============================================================
-- 表名: sys_menu
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 菜单表，支持四种类型：目录(Catalog)、菜单(Menu)、外链(Link)、按钮(Button)。
-- type 字段区分类型，perm 字段存储权限标识（如 sys:user:add）用于接口鉴权。
-- 按钮类型不渲染路由，仅用于前端按钮显示控制和后端权限校验。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_menu`;
CREATE TABLE `sys_menu`
(
    `id`          bigint                                                       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `parent_id`   bigint                                                       NOT NULL COMMENT '父菜单ID',
    `tree_path`   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '父节点ID路径',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '菜单名称',
    `type`        tinyint                                                      NOT NULL COMMENT '菜单类型(1:菜单 2:目录 3:外链 4:按钮)',
    `path`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '路由路径(浏览器地址栏路径)',
    `component`   varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '组件路径(vue页面完整路径，省略.vue后缀)',
    `perm`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '权限标识',
    `visible`     tinyint                                                      NOT NULL DEFAULT 1 COMMENT '显示状态(1:显示;0:隐藏)',
    `status`      tinyint                                                      NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `sort`        int                                                          NULL DEFAULT 0 COMMENT '排序',
    `icon`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '菜单图标',
    `redirect`    varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '跳转路径',
    `always_show` tinyint                                                      NULL DEFAULT NULL COMMENT '【目录】只有一个子路由是否始终显示(1:是 0:否)',
    `keep_alive`  tinyint                                                      NULL DEFAULT NULL COMMENT '【菜单】是否开启页面缓存(1:是 0:否)',
    `deleted`     tinyint                                                      NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                     NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                       NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                       NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '菜单管理'
  ROW_FORMAT = DYNAMIC;
