/*
* Pei 图像去雾系统数据库(MySQL8.x)
* @author earthyzinc
*/

-- ----------------------------
-- 1. 创建数据库
-- ----------------------------
CREATE DATABASE IF NOT EXISTS dehaze DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_general_ci;


-- ----------------------------
-- 2. 创建表 && 数据初始化
-- ----------------------------
use dehaze;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;


-- ----------------------------
-- Table structure for sys_dept
-- ----------------------------
DROP TABLE IF EXISTS `sys_dept`;
CREATE TABLE `sys_dept`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NOT NULL DEFAULT '' COMMENT '部门名称',
    `parent_id`   bigint                                                        NOT NULL DEFAULT 0 COMMENT '父节点id',
    `tree_path`   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL     DEFAULT '' COMMENT '父节点id路径',
    `sort`        int                                                           NULL     DEFAULT 0 COMMENT '显示顺序',
    `status`      tinyint                                                       NOT NULL DEFAULT 1 COMMENT '状态(1:正常;0:禁用)',
    `deleted`     tinyint                                                       NULL     DEFAULT 0 COMMENT '逻辑删除标识(1:已删除;0:未删除)',
    `create_time` datetime                                                      NULL     DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                      NULL     DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 171
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '部门表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_dept
-- ----------------------------
insert into sys_dept (id, name, parent_id, tree_path, sort, status, deleted, create_time, update_time, create_by,
                      update_by)
values (1, '重庆邮电大学', 0, '0', 1, 1, 0, null, '2024-11-13 14:39:21', 1, 2),
       (2, '软件工程学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:32', 2, 2),
       (3, '计算机学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:42', 2, 2);

-- ----------------------------
-- Table structure for sys_dict
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `type_code`   varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT NULL COMMENT '字典类型编码',
    `name`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT '' COMMENT '字典项名称',
    `value`       varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT '' COMMENT '字典项值',
    `sort`        int                                                           NULL DEFAULT 0 COMMENT '排序',
    `status`      tinyint                                                       NULL DEFAULT 0 COMMENT '状态(1:正常;0:禁用)',
    `defaulted`   tinyint                                                       NULL DEFAULT 0 COMMENT '是否默认(1:是;0:否)',
    `remark`      varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '备注',
    `create_time` datetime                                                      NULL DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 69
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典数据表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_dict
-- ----------------------------
insert into sys_dict (id, type_code, name, value, sort, status, defaulted, remark, create_time, update_time)
values (1, 'gender', '男', '1', 1, 1, 0, null, '2019-05-05 13:07:52', '2022-06-12 23:20:39'),
       (2, 'gender', '女', '2', 2, 1, 0, null, '2019-04-19 11:33:00', '2019-07-02 14:23:05'),
       (3, 'gender', '未知', '0', 1, 1, 0, null, '2020-10-17 08:09:31', '2020-10-17 08:09:31');

-- ----------------------------
-- Table structure for sys_dict_type
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict_type`;
CREATE TABLE `sys_dict_type`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键 ',
    `name`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT '' COMMENT '类型名称',
    `code`        varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT '' COMMENT '类型编码',
    `status`      tinyint(1)                                                    NULL DEFAULT 0 COMMENT '状态(0:正常;1:禁用)',
    `remark`      varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '备注',
    `create_time` datetime                                                      NULL DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `type_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 89
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典类型表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_dict_type
-- ----------------------------
INSERT INTO `sys_dict_type`
VALUES (1, '性别', 'gender', 1, NULL, '2019-12-06 19:03:32', '2022-06-12 16:21:28');

-- ----------------------------
-- Table structure for sys_menu
-- ----------------------------
DROP TABLE IF EXISTS `sys_menu`;
CREATE TABLE `sys_menu`
(
    `id`          bigint                                                       NOT NULL AUTO_INCREMENT,
    `parent_id`   bigint                                                       NOT NULL COMMENT '父菜单ID',
    `tree_path`   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT NULL COMMENT '父节点ID路径',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL DEFAULT '' COMMENT '菜单名称',
    `type`        tinyint                                                      NOT NULL COMMENT '菜单类型(1:菜单 2:目录 3:外链 4:按钮)',
    `path`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT '' COMMENT '路由路径(浏览器地址栏路径)',
    `component`   varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT NULL COMMENT '组件路径(vue页面完整路径，省略.vue后缀)',
    `perm`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT NULL COMMENT '权限标识',
    `visible`     tinyint(1)                                                   NOT NULL DEFAULT '1' COMMENT '显示状态(1-显示;0-隐藏)',
    `sort`        int                                                                   DEFAULT '0' COMMENT '排序',
    `icon`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci          DEFAULT '' COMMENT '菜单图标',
    `redirect`    varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT NULL COMMENT '跳转路径',
    `create_time` datetime                                                              DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                              DEFAULT NULL COMMENT '更新时间',
    `always_show` tinyint                                                               DEFAULT NULL COMMENT '【目录】只有一个子路由是否始终显示(1:是 0:否)',
    `keep_alive`  tinyint                                                               DEFAULT NULL COMMENT '【菜单】是否开启页面缓存(1:是 0:否)',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 103
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  ROW_FORMAT = DYNAMIC COMMENT ='菜单管理';

-- ----------------------------
-- Records of sys_menu
-- ----------------------------

-- ----------------------------
-- Records of sys_menu
-- ----------------------------
insert into sys_menu (id, parent_id, tree_path, name, type, path, component, perm, visible, sort, icon, redirect,
                      create_time, update_time, always_show, keep_alive)
values (1, 0, '0', '系统管理', 2, '/system', 'Layout', null, 1, 1, 'system', '/system/user', '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, null),
       (2, 1, '0,1', '用户管理', 1, 'user', 'system/user/index', null, 1, 1, 'user', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (3, 1, '0,1', '角色管理', 1, 'role', 'system/role/index', null, 1, 2, 'role', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (4, 1, '0,1', '菜单管理', 1, 'menu', 'system/menu/index', null, 1, 3, 'menu', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (5, 1, '0,1', '部门管理', 1, 'dept', 'system/dept/index', null, 1, 4, 'tree', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (6, 1, '0,1', '字典管理', 1, 'dict', 'system/dict/index', null, 1, 5, 'dict', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (31, 2, '0,1,2', '用户新增', 4, '', null, 'sys:user:add', 1, 1, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (32, 2, '0,1,2', '用户编辑', 4, '', null, 'sys:user:edit', 1, 2, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (33, 2, '0,1,2', '用户删除', 4, '', null, 'sys:user:delete', 1, 3, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (70, 3, '0,1,3', '角色新增', 4, '', null, 'sys:role:add', 1, 1, '', null, '2023-05-20 23:39:09',
        '2023-05-20 23:39:09', null, null),
       (71, 3, '0,1,3', '角色编辑', 4, '', null, 'sys:role:edit', 1, 2, '', null, '2023-05-20 23:40:31',
        '2023-05-20 23:40:31', null, null),
       (72, 3, '0,1,3', '角色删除', 4, '', null, 'sys:role:delete', 1, 3, '', null, '2023-05-20 23:41:08',
        '2023-05-20 23:41:08', null, null),
       (73, 4, '0,1,4', '菜单新增', 4, '', null, 'sys:menu:add', 1, 1, '', null, '2023-05-20 23:41:35',
        '2023-05-20 23:41:35', null, null),
       (74, 4, '0,1,4', '菜单编辑', 4, '', null, 'sys:menu:edit', 1, 3, '', null, '2023-05-20 23:41:58',
        '2023-05-20 23:41:58', null, null),
       (75, 4, '0,1,4', '菜单删除', 4, '', null, 'sys:menu:delete', 1, 3, '', null, '2023-05-20 23:44:18',
        '2023-05-20 23:44:18', null, null),
       (76, 5, '0,1,5', '部门新增', 4, '', null, 'sys:dept:add', 1, 1, '', null, '2023-05-20 23:45:00',
        '2023-05-20 23:45:00', null, null),
       (77, 5, '0,1,5', '部门编辑', 4, '', null, 'sys:dept:edit', 1, 2, '', null, '2023-05-20 23:46:16',
        '2023-05-20 23:46:16', null, null),
       (78, 5, '0,1,5', '部门删除', 4, '', null, 'sys:dept:delete', 1, 3, '', null, '2023-05-20 23:46:36',
        '2023-05-20 23:46:36', null, null),
       (79, 6, '0,1,6', '字典类型新增', 4, '', null, 'sys:dict_type:add', 1, 1, '', null, '2023-05-21 00:16:06',
        '2023-05-21 00:16:06', null, null),
       (81, 6, '0,1,6', '字典类型编辑', 4, '', null, 'sys:dict_type:edit', 1, 2, '', null, '2023-05-21 00:27:37',
        '2023-05-21 00:27:37', null, null),
       (84, 6, '0,1,6', '字典类型删除', 4, '', null, 'sys:dict_type:delete', 1, 3, '', null, '2023-05-21 00:29:39',
        '2023-05-21 00:29:39', null, null),
       (85, 6, '0,1,6', '字典数据新增', 4, '', null, 'sys:dict:add', 1, 4, '', null, '2023-05-21 00:46:56',
        '2023-05-21 00:47:06', null, null),
       (86, 6, '0,1,6', '字典数据编辑', 4, '', null, 'sys:dict:edit', 1, 5, '', null, '2023-05-21 00:47:36',
        '2023-05-21 00:47:36', null, null),
       (87, 6, '0,1,6', '字典数据删除', 4, '', null, 'sys:dict:delete', 1, 6, '', null, '2023-05-21 00:48:10',
        '2023-05-21 00:48:20', null, null),
       (88, 2, '0,1,2', '重置密码', 4, '', null, 'sys:user:password:reset', 1, 4, '', null, '2023-05-21 00:49:18',
        '2024-04-28 00:38:22', null, null),
       (105, 2, '0,1,2', '用户查询', 4, '', null, 'sys:user:query', 1, 0, '', null, '2024-04-28 00:37:34',
        '2024-04-28 00:37:34', 0, 0),
       (106, 2, '0,1,2', '用户导入', 4, '', null, 'sys:user:import', 1, 5, '', null, '2024-04-28 00:39:15',
        '2024-04-28 00:39:15', null, null),
       (107, 2, '0,1,2', '用户导出', 4, '', null, 'sys:user:export', 1, 6, '', null, '2024-04-28 00:39:43',
        '2024-04-28 00:39:43', null, null);

-- --------------------
-- Table structure for sys_role
-- ----------------------------
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
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `name` (`name` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 128
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '角色表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_role
-- ----------------------------
insert into sys_role (id, name, code, sort, status, data_scope, deleted, create_time, update_time)
values (1, '超级管理员', 'ROOT', 1, 1, 0, 0, '2021-05-21 14:56:51', '2018-12-23 16:00:00'),
       (2, '系统管理员', 'ADMIN', 2, 1, 1, 0, '2021-03-25 12:39:54', null),
       (3, '访问游客', 'GUEST', 3, 1, 2, 0, '2021-05-26 15:49:05', '2019-05-05 16:00:00'),
       (4, '系统管理员1', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:28'),
       (5, '系统管理员2', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (6, '系统管理员3', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (7, '系统管理员4', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (8, '系统管理员5', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (9, '系统管理员6', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (10, '系统管理员7', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (11, '系统管理员8', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36'),
       (12, '系统管理员9', 'ADMIN1', 2, 1, 1, 1, '2021-03-25 12:39:54', '2024-11-13 14:40:36');

-- ----------------------------
-- Table structure for sys_role_menu
-- ----------------------------
DROP TABLE IF EXISTS `sys_role_menu`;
CREATE TABLE `sys_role_menu`
(
    `role_id` bigint NOT NULL COMMENT '角色ID',
    `menu_id` bigint NOT NULL COMMENT '菜单ID'
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '角色和菜单关联表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_role_menu
-- ----------------------------
insert into dehaze.sys_role_menu (role_id, menu_id)
values (9, 1),
       (9, 3),
       (9, 70),
       (9, 72),
       (3, 1),
       (3, 2),
       (3, 31),
       (3, 32),
       (3, 33),
       (3, 88),
       (3, 3),
       (3, 70),
       (3, 71),
       (3, 72),
       (2, 1),
       (2, 2),
       (2, 105),
       (2, 31),
       (2, 32),
       (2, 33),
       (2, 88),
       (2, 106),
       (2, 107),
       (2, 3),
       (2, 70),
       (2, 71),
       (2, 72),
       (2, 4),
       (2, 73),
       (2, 75),
       (2, 74),
       (2, 5),
       (2, 76),
       (2, 77),
       (2, 78),
       (2, 6),
       (2, 79),
       (2, 81),
       (2, 84),
       (2, 85),
       (2, 86),
       (2, 87),
       (2, 40),
       (2, 41),
       (2, 26),
       (2, 102),
       (2, 30),
       (2, 20),
       (2, 21),
       (2, 22),
       (2, 23),
       (2, 24),
       (2, 36),
       (2, 37),
       (2, 38),
       (2, 39),
       (2, 95),
       (2, 89),
       (2, 97),
       (2, 90),
       (2, 91);

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user`
(
    `id`          int                                                           NOT NULL AUTO_INCREMENT,
    `username`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT NULL COMMENT '用户名',
    `nickname`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT NULL COMMENT '昵称',
    `gender`      tinyint(1)                                                    NULL DEFAULT 1 COMMENT '性别((1:男;2:女))',
    `password`    varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '密码',
    `dept_id`     int                                                           NULL DEFAULT NULL COMMENT '部门ID',
    `avatar`      TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         NULL COMMENT '用户头像',
    `mobile`      varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL DEFAULT NULL COMMENT '联系方式',
    `status`      tinyint(1)                                                    NULL DEFAULT 1 COMMENT '用户状态((1:正常;0:禁用))',
    `email`       varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '用户邮箱',
    `deleted`     tinyint(1)                                                    NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                      NULL DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `login_name` (`username` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 288
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '用户信息表'
  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_user
-- ----------------------------
insert into dehaze.sys_user (id, username, nickname, gender, password, dept_id, avatar, mobile, status, email, deleted,
                             create_time, update_time)
values (1, 'root', '有来技术', 0, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', null,
        '', '17621590365', 1,
        'youlaitech@163.com', 0, null, null),
       (2, 'admin', '武沛鑫', 1, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', 1,
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/4QAuRXhpZgAATU0AKgAAAAgAAkAAAAMAAAABAAAAAEABAAEAAAABAAAAAAAAAAD/2wBDAAoHBwkHBgoJCAkLCwoMDxkQDw4ODx4WFxIZJCAmJSMgIyIoLTkwKCo2KyIjMkQyNjs9QEBAJjBGS0U+Sjk/QD3/2wBDAQsLCw8NDx0QEB09KSMpPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT3/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCxD4Xs4yCxkcjtnGa0obKCEAInPuSasVDeXC2dtJKwztHA9T2FDk92xRjd2Rm6vMhlW2THynzJGznb6U7SodkbTkHMvI9gKz4onuZxG53SSnfKw7D0/pW6AAoAwABgCuKD55Ob22R6KiqcVFb9Re1J7UtFakB+FFFB6UAFY13F9mvyAP3dxlhnsw5I/Hg/nWkb60jcq1zAr/3TIoP86h1GL7ZZFrdlaRGEkZByCw7Z9xkfjWdWHPFoum2mUAWhuYrlBueIk7R/EpGGH1xyPcV0EM0dzCssLBo2+YEf5/SuejkWWMSL91hkZqexkaG9EQcrFcE5A7OBwR6ZAP5CufC1nF+zYYmjzLmW6N75v8iiovs69yxPqWNFemeeSce1ZGsS75oYB91QZG/pWo8Z6qxRvzB/CsHUSwurjdt3kBRj8q58Q2oaddDbDJOav0J9ODiMyIoMsxzuPRV7Z/wqZ7GeTltQuVJ7RhEA/Q/zqzDH5UMaAfdUCnmRU+8yr9TRGKUUkdUm27meljqEM0bJqbSRBvnSaJWJHswxzWiCOlV7m9gt4y7zRLt5wXC5qaNxIgdSCrAEEdxVEt3HVSvtJh1GRWuXnaNVx5IlKo3uQOtXfrVa8uWtnVvJmkTH/LNd3PvyMUAnbUdFp9pCgSG1gRB2Ea/4UptkHzQgROOjIMD8R0NZMvim3t544poZ1d2KqpTO4/gTVqHxBp8zhDKUkborqw/mOaHFrdCU03oyvcQm1vCuAI58uuOgbuv49fzqOYsIi6Z8xCHX6qc/0rYurZby2MbMV6MrDqrDoRWPG7ESLMoEkTFXA6ZHp7EYP4159em4TU0dMHzRszo45d8SOhG1lDD8RRVbQ5PN0Oyfd1iX+VFekpHA4Fvr1rn75y/iSOIdCoJ/DmtsvJFGXcx4QFieRWDag3GqxXMn+sbe2PQdh+VRWa0T6seGi7t9ka1zDJMgVLiSH1KAZP4kHFY994aa/wAhr6dVLA5Lsxx6dQP0reoqk2tjVxT3M+z0Szs8MsSvIOjOAdp9hjitAelHeo5ZWjO0RPIe+zHH5mhtvcaSSsiTOaKT5tnA59DTITKUPnoqyBiPkOQR2NIBzxRyfejVj6kA0JGiD5VVfoMU6jNAgrG1GNhfzBeDLEjfjkqf0xWzWPq7ql4Ds3P5B2tnG3DE5/SsqkOeLSNIOzLOmTeRYJEifIhZV+gY0Vc0WIPpMD4Pzgv+ZJ/rRQk7GbkrlXWLjOLNDyxDSY7L6fjVS14v4j6qwqKNCinP3mOSfWnxnZdQN/tY/OuR1vaVl2N4U+SFuptUUnNHWvQMRagur2GzAMz4LfdUDJP0FNv71bKHIw0rcKp7+59q5d7hppGkAaV2PzOTgH6e304oOqhh3U1eiOifWbaMAhmbd1AH3frVuG5juU3QuGHt1H1rkd8x/wCWaj6t/wDWp0VzPbOJApBHdDn8x3oN54RW0ep2GKKpadqcV8gAZVlH3lz+oq7Qee4tOzDH5Vl3QEmshXUtH9mK/id39Aa08VRvNqSXMpUF0ttyn0+8KQLcv6WR/ZVp5bfJ5KY/75FFYdnrp0qzis2hDmIbd3t2/IYH4UUvbIz9k+wvWmucGNv7rA06myAlCq9f8K8ikm5pLe53TaUW3sbgNGc0y3kEsEbg5ytP69K9k5TJv9FkupmlE5bd1V+mPT6VANFusdYuOnJreooNo4icFZMwk0S5P3miX8SasxaDEB++ld29FO0VqUUBLEVGtyvbWFtZktDEquerYyT+NWMUUc0GLberDnvWPfFpdQmQPiMIiMoH3urYz+IrXJAGScDuTWGkhleScjCzuWX6dB+gFY11LkbiOEkpJPqOKI5yygk98UUUV5N5dzssgp0aMX3AcKOauppmfvSn8BUyWcds4xltwwxJr18NgZwmpStoeXiMZCUHGPUpaVL5Rkt2PAYlfpWkOBVSWxCOXUnnp7GpIZmJ8uRTu/vDvXTUg0x0ailFa6k5+9S59ail80ENGFYD+E96hF6o4mikU/TIrM2sWvSlqAX0H/PQD6g0G9g/56rQGpPRVf7Yh4RXf6CnySsEAVMyN91T0HuaA1KmsSN9imiiYhinzMP4QeMfUk/zqtKNkYVV4UjgdgKs3sP+hiJSZJWlRmPdsMCfw4qf7L5dnMz4MjIc+3tW8KfNBrucWIqWnG3QzaKma2mUkFeaK8d4Sr/Kz0vrNPujaqOb/VmiivpD54eOUH0qtOAhyvB9qKKzqbGlLcWJ2PBPFPNFFcMtz1o7IBGhxlF/KkMaL0RfyoopFCSEog28VEpMg+Yk/jRRWlM565ajjRPuqBUdz/x7N7gfzFFFdx5nUmNFFFMD/9k=',
        '18537958917', 1,
        'w1066365803@163.com', 0, '2019-10-10 13:41:22', '2024-11-13 14:40:05'),
       (3, 'test', '测试小用户', 1, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', 3,
        '', '17621210366', 1,
        'youlaitech@163.com', 0, '2021-06-05 01:31:29', '2021-06-05 01:31:29'),
       (287, '123', '123', 1, '$2a$10$mVoBVqm1837huf7kcN0wS.GVYKEFv0arb7GvzfFXoTyqDlcRzT.6i', 1, '', null, 1, null, 1,
        '2023-05-21 14:11:19', '2023-05-21 14:11:25');

-- ----------------------------
-- Table structure for sys_user_role
-- ----------------------------
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

-- ----------------------------
-- Records of sys_user_role
-- ----------------------------
insert into dehaze.sys_user_role (user_id, role_id)
values (1, 1),
       (2, 2),
       (3, 3),
       (287, 2);

DROP TABLE IF EXISTS `sys_dataset`;
CREATE TABLE `sys_dataset`
(
    `id`          bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '数据集ID',
    `parent_id`   bigint                                                         NOT NULL DEFAULT 0 COMMENT '父数据集ID',
    `type`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci   NOT NULL DEFAULT '' COMMENT '数据集类型',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci   NOT NULL DEFAULT '' COMMENT '数据集名称',
    `description` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL     DEFAULT '' COMMENT '数据集描述',
    `path`        varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NOT NULL DEFAULT '' COMMENT '存储位置',
    `size`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL     DEFAULT '' COMMENT '占用空间大小',
    `total`       int                                                            NULL     DEFAULT 0 COMMENT '数据项数量（简单理解为图片数量）',
    `status`      tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用；0:禁用)',
    `deleted`     tinyint                                                        NULL     DEFAULT 0 COMMENT '逻辑删除标识(1:已删除;0:未删除)',
    `create_time` datetime                                                       NULL     DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                       NULL     DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint                                                         NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                         NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '数据集表'
  ROW_FORMAT = DYNAMIC;

insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (1, 0, '图像去雾', 'DENSE-HAZE',
        'DENSE-HAZE 引入了一种新的去雾数据集，以浓密均匀的雾霾场景为特征。该数据集包含 55对真实的浓雾图像和各种室外场景的相应无雾图像。这些朦胧图像是通过专业雾霾机器生成的真实雾霾记录的。生成的浓雾图像几乎难以辨别图像中原来存在的物体，与常规数据集相比去雾难度非常大。',
        'Dense-Haze', '234.74 MB', 110, 1, 0, '2024-11-11 19:29:49', '2024-11-11 19:29:49', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (2, 0, '图像去雾', 'O-HAZE',
        'O-haze 数据集是由CVLab实验室在2016年发布的，主要用于评估和测试图像去雾算法的性能。该数据集包含了合成的有雾图像和相应的清晰图像对，这些图像都是基于真实的户外场景生成的。包含45对户外场景的有雾和清晰图像',
        'O-HAZE', '547.85 MB', 90, 1, 0, '2024-11-11 19:36:32', '2024-11-11 19:36:32', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (3, 0, '图像去雾', 'I-HAZE',
        'I-haze 数据集也是由CVLab实验室在2016年发布的，与O-haze数据集类似，它主要用于评估和测试图像去雾算法的性能。不过，I-haze数据集的特点在于其图像更接近实际的室内场景。包含35对有雾和相应的无雾室内图像。',
        'I-HAZE', '312.99 MB', 60, 1, 0, '2024-11-11 19:37:12', '2024-11-11 19:37:12', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (4, 0, '图像去雾', 'NH-HAZE',
        'NH-HAZE数据集旨在解决图像去雾领域中的一个重要问题：缺乏真实世界的非均匀雾度图像作为参考数据。许多现实场景中的雾并不均匀分布，因此 NH-HAZE 提供了一组真实的非均匀雾图像和相应的无雾图像对。NH-HAZE 数据集中的非均匀雾度是通过专业的雾发生器模拟真实雾天条件而引入的。是一个更具挑战性和现实性的去雾数据集。',
        'NTIRE', '1.06 GB', 241, 1, 0, '2024-11-11 19:39:24', '2024-11-11 19:39:24', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (5, 4, '图像去雾', 'NH-HAZE-20',
        '在2020年，NH-HAZE数据集被用于CVPR NTIRE（New Trends in Image Restoration and Enhancement）研讨会下的图像去雾在线挑战赛中1。这是首个包含55对外部拍摄的真实有雾和对应的无雾图像的数据集，这些图像是使用专业雾生成器在高保真的条件下拍摄的，以模拟真实的非均匀雾霾环境。NH-HAZE 2020的数据集为研究人员提供了评估去雾算法性能的机会，并且由于其现实性，对于开发更加鲁棒的去雾解决方案具有重要意义',
        'NH-HAZE-2020', '316.96 MB', 110, 1, 0, '2024-11-11 19:39:51', '2024-11-11 19:39:51', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (6, 4, '图像去雾', 'NH-HAZE-21',
        '到了2021年，NTIRE挑战赛继续进行，这次的非均匀去雾挑战基于扩展后的NH-HAZE数据集，增加了额外35对真实户外拍摄的无雾和非均匀有雾图像。这个扩大的数据集被称为NH-Haze2，它进一步增强了数据集的多样性和复杂度，为参与者提供了更广泛的测试平台来验证他们的算法。此外，在这次挑战中还加入了其他小规模的真实世界数据集如DENSE-HAZE等，用以对比不同方法的效果。',
        'NH-HAZE-2021', '151.36 MB', 50, 1, 0, '2024-11-11 19:40:08', '2024-11-11 19:40:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (7, 4, '图像去雾', 'NH-HAZE-23',
        '至2023年，NTIRE举办了一次高清非均质去雾挑战赛，这次比赛采用了名为HD-NH-HAZE的新数据集。HD-NH-HAZE包含了50对高清分辨率的户外图像，其中一半是带有非均匀雾霾的图像，另一半则是同一场景的无雾霾图像。这个数据集的引入标志着单张图像去雾领域的一个重要进展，因为它不仅提高了图像的质量标准，而且也推动了去雾技术向着处理更高分辨率图像的方向发展。参赛者们提出的方法在此数据集上进行了客观评估，以便更好地衡量它们在处理实际场景中的表现',
        'NH-HAZE-2023', '618.19 MB', 80, 1, 0, '2024-11-11 19:40:19', '2024-11-11 19:40:19', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (8, 0, '图像去雾', 'RESIDE',
        'RESIDE（Realistic Synthetic and Indoor-Outdoor DEhazing）数据集是由北京大学和微软亚洲研究院在2017年联合发布的，旨在为图像去雾研究提供一个大规模、多样化的基准数据集。RESIDE 数据集不仅包含合成的有雾图像和对应的清晰图像，还包含了一些真实世界中的有雾图像，使其成为图像去雾领域最全面的数据集之一。',
        'RESIDE', '19.01 GB', 117915, 1, 0, '2024-11-11 19:41:55', '2024-11-11 19:41:55', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (9, 8, '图像去雾', 'ITS',
        '室内训练集(ITS) 是RESIDE数据集中的一部分，主要用于算法的训练阶段。ITS包含13,990张由清晰图像生成的合成模糊图像，这些清晰图像是从现有的室内深度数据集NYU2和米德尔伯里立体数据库中选取的1,399张图像。对于每一张清晰图像，通过在不同参数设置下（例如大气光A和散射系数β）生成10张模糊图像。这些参数的设定使得生成的模糊图像能够模拟多种不同的雾霾情况。具体来说，大气光A的值在[0.7, 1.0]之间均匀随机选择，而β则在[0.6, 1.8]之间均匀随机选择。最终，这13,990张图像中的13,000张被用于训练，剩下的990张作为验证集',
        'RESIDE/ITS', '4.74 GB', 29379, 1, 0, '2024-11-11 19:42:34', '2024-11-11 19:42:34', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (10, 8, '图像去雾', 'OTS',
        '室外训练集(OTS) 是RESIDE-beta部分的数据集，旨在提高对室外环境下的去雾性能。OTS使用了2061张来自北京实时天气的真实室外图像，通过估计每张图像的深度信息后，根据一系列指定的大气散射系数β值（如0.04, 0.06, 0.08, 0.1, 0.15, 0.95, 1等）来合成模糊图像。最终，总共合成了72,135张户外模糊图像。这套新的图像被称为户外训练集（OTS），由成对的干净的户外图像和生成的模糊图像组成，以供算法训练使用',
        'RESIDE/OTS', '12.86 GB', 78318, 1, 0, '2024-11-11 19:42:54', '2024-11-11 19:42:54', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (11, 8, '图像去雾', 'SOTS',
        '综合目标测试集(SOTS) 用于客观评价去雾算法的表现。SOTS包含了两种类型的图像：一种是从NYU2数据集中挑选出的500个室内场景图像（与ITS中的训练图像不重叠），并按照与ITS相同的流程来合成模糊图像；另一种则是从实际拍摄的室外场景中收集的图像。SOTS不仅包括了合成的模糊图像，还提供了对应的真实无雾图像，这样可以用来计算去雾后的图像质量指标，如PSNR、SSIM等',
        'RESIDE/SOTS', '416.3 MB', 1542, 1, 0, '2024-11-12 22:21:07', '2024-11-12 22:21:07', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (12, 11, '图像去雾', 'indoor', 'SOTS Indoor (室内): 从NYU2中选择了500个室内图像，用以生成相应的模糊图像。',
        'RESIDE/SOTS/indoor', '170 MB', 550, 1, 0, '2024-11-12 22:21:37', '2024-11-12 22:21:37', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (13, 11, '图像去雾', 'outdoor',
        'SOTS Outdoor (室外): 包括了真实的室外模糊图像及其对应的清晰图像，用于更接近实际应用的评估。',
        'RESIDE/SOTS/outdoor', '246.3 MB', 992, 1, 0, '2024-11-12 22:21:57', '2024-11-12 22:21:57', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (14, 0, '图像去雾', 'RESIDE-6k',
        '虽然没有直接提到名为RESIDE-6k的数据集，但我们可以假设这可能是一个包含大约6000张图像的RESIDE数据集的一个子集。如果这是对RESIDE数据集的特定版本，则它可能专注于一个特定的场景（室内或室外）或者用于特定目的（比如训练或测试）。然而，由于没有具体的信息，我们无法确定其确切组成。通常，这样的数据集会包含成对的清晰和模糊图像，以便于模型学习如何从模糊图像恢复清晰图像。',
        'RESIDE-6k', '1.52 GB', 14000, 1, 0, '2024-11-12 22:22:42', '2024-11-12 22:22:42', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (15, 14, '图像去雾', 'RESIDE-6k-train',
        'RESIDE-6k 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-6k/train', '1,021.78 MB',
        12000, 1, 0, '2024-11-12 22:23:05', '2024-11-12 22:23:05', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (16, 14, '图像去雾', 'RESIDE-6k-test',
        'RESIDE-6k  测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-6k/test', '532.3 MB',
        2000, 1, 0, '2024-11-12 22:23:16', '2024-11-12 22:23:16', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (17, 0, '图像去雾', 'RESIDE-IN',
        '这个名称可能指的是RESIDE数据集中专注于室内场景的部分。结合RESIDE数据集的描述，我们可以合理推测RESIDE-IN可能主要包含了ITS（Indoor Training Set），即室内训练集。该集合包括了13,990个合成的模糊图像，这些图像是基于NYU2和米德尔伯里立体数据库中的1,399个清晰室内图像生成的1。此外，SOTS（Synthetic Objective Testing Set）中的部分室内图像也可能被包含在内，用于评估算法性能。',
        'RESIDE-IN', '8.74 GB', 28980, 1, 0, '2024-11-12 22:23:48', '2024-11-12 22:23:48', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (18, 17, '图像去雾', 'RESIDE-IN-train',
        'RESIDE-IN 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-IN/train', '8.36 GB', 27980,
        1, 0, '2024-11-12 22:24:10', '2024-11-12 22:24:10', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (19, 17, '图像去雾', 'RESIDE-IN-test',
        'RESIDE-IN 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-IN/test', '392.11 MB',
        1000, 1, 0, '2024-11-12 22:25:09', '2024-11-12 22:25:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (20, 0, '图像去雾', 'RESIDE-OUT',
        '同样，RESIDE-OUT可能是指RESIDE数据集中专注于室外场景的部分。这意味着它可能主要由OTS（Outdoor Training Set）构成，该集合包括72,135张合成的户外模糊图像，这些图像是基于北京实时天气的真实室外图像生成的2。SOTS中的一部分室外图像也可能会被纳入其中，用于测试目的。',
        'RESIDE-OUT', '83.03 GB', 628480, 1, 0, '2024-11-12 22:26:09', '2024-11-12 22:26:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (21, 20, '图像去雾', 'RESIDE-OUT-train',
        'RESIDE-OUT 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-OUT/train', '82.89 GB',
        627480, 1, 0, '2024-11-12 22:26:47', '2024-11-12 22:26:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (22, 20, '图像去雾', 'RESIDE-OUT-test',
        'RESIDE-OUT 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-OUT/test',
        '140.19 MB', 1000, 1, 0, '2024-11-12 22:27:08', '2024-11-12 22:27:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (23, 0, '图像去雾', 'RSHAZE',
        'REHAZE数据集是专为图像去雾研究设计的，旨在提供更真实的雾霾条件下的图像。它由苏黎世联邦理工大学等机构发布，包含有雾和无雾图像对，这些图像是在受控环境中使用专业设备拍摄的，以模拟不同的雾霾条件。不过，具体的细节（如图像数量、场景类型等）需要查阅原始论文或官方发布页面来获取准确信息。',
        'RSHAZE', '40.41 GB', 108000, 1, 0, '2024-11-12 22:28:28', '2024-11-12 22:28:28', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (24, 23, '图像去雾', 'RSHAZE-train', 'REHAZE 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。',
        'RSHAZE/train', '38.39 GB', 102600, 1, 0, '2024-11-12 22:28:47', '2024-11-12 22:28:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, description, path, size, total, status, deleted, create_time,
                         update_time, create_by, update_by)
values (25, 23, '图像去雾', 'RSHAZE-test', 'REHAZE 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。',
        'RSHAZE/test', '2.02 GB', 5400, 1, 0, '2024-11-12 22:28:54', '2024-11-12 22:28:54', 2, 2);

DROP TABLE IF EXISTS `sys_file`;
CREATE TABLE `sys_file`
(
    `id`          int                                                           NOT NULL AUTO_INCREMENT COMMENT '文件id',
    `type`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci          DEFAULT NULL COMMENT '文件类型',
    `url`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci                  DEFAULT NULL COMMENT '文件url',
    `name`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件原始名',
    `object_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件存储名',
    `size`        varchar(100)                                                  NOT NULL DEFAULT '0' COMMENT '文件大小',
    `path`        varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件路径',
    `md5`         char(32) unique                                               NOT NULL COMMENT '文件的MD5值，用于比对文件是否相同',
    `create_time` datetime                                                      NOT NULL COMMENT '创建时间',
    `update_time` datetime                                                               DEFAULT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `md5_key` (`md5` ASC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='文件表';

DROP TABLE IF EXISTS `sys_algorithm`;
CREATE TABLE `sys_algorithm`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '模型id',
    `parent_id`   bigint           DEFAULT 0 COMMENT '模型的父id',
    `type`        varchar(100)     DEFAULT '' COMMENT '模型类型',
    `name`        varchar(64) NOT NULL COMMENT '模型名称',
    `path`        varchar(255)     DEFAULT '' COMMENT '模型存储路径',
    `size`        varchar(100)     DEFAULT NULL COMMENT '模型大小',
    `import_path` varchar(255)     DEFAULT NULL COMMENT '模型代码导入路径',
    `description` varchar(2048)    DEFAULT NULL COMMENT '针对该模型的详细描述',
    `status`      tinyint(1)       DEFAULT 1 COMMENT '状态(1:启用；0:禁用)',
    `create_time` datetime         DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime         DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint      NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint      NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法模型表';

insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (1, 0, '图像去雾', 'AECR-NET', 'AECR-Net/NH_train.pk', '35.1 MB', 'algorithm.AECRNet.run',
        'AECRNet 是一种深度学习模型，专门用于图像去雾任务。该模型由清华大学和微软亚洲研究院的研究人员在2019年提出，旨在解决传统去雾方法中存在的边缘模糊和细节丢失问题。AECRNet 通过引入对抗生成网络（GAN）和边缘保持机制，实现了高质量的去雾效果。',
        1, '2024-11-11 20:00:28', '2024-11-11 20:00:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (2, 0, '图像去雾', 'AODNet', 'AODNet/dehazer.pth', '8.41 KB', 'algorithm.AODNet.run',
        'AODNet (All-in-One Dehazing Network) 是一种用于图像去雾的深度学习模型，由Yuan et al. 在2018年提出。传统的图像去雾方法通常依赖于大气散射模型以及一些先验知识，如暗通道先验等，这些方法虽然在某些情况下能够取得较好的效果，但是往往计算复杂度较高，且对于不同的环境条件适应性较差。AODNet旨在解决这些问题，提供一个更加高效和鲁棒的解决方案。',
        1, '2024-11-11 23:52:37', '2024-11-11 23:52:37', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (3, 0, '图像去雾', 'C2PNet', 'C2PNet/ITS.pkl', '35.98 MB', 'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:51:50', '2024-11-12 22:51:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (4, 3, '图像去雾', '室内去雾（ITS）', 'C2PNet/ITS.pkl', '35.98 MB', 'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:52:15', '2024-11-12 22:52:15', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (5, 3, '图像去雾', '室外去雾（OTS）', 'C2PNet/OTS.pkl', '39.51 MB', 'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:52:25', '2024-11-12 22:52:25', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (6, 0, '图像去雾', 'CMFNet', 'CMFNet', null, 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:00', '2024-11-13 11:30:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (7, 6, '图像去雾', '去雾模型', 'CMFNet/dehaze_I_OHaze_CMFNet.pth', '197.53 MB', 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:16', '2024-11-13 11:30:16', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (8, 6, '图像去模糊', '去模糊模型', 'CMFNet/deblur_GoPro_CMFNet.pth', '197.53 MB', 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:26', '2024-11-13 11:30:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (9, 6, '图像去雨', '去雨模型', 'CMFNet/deraindrop_DeRainDrop_CMFNet.pth', '197.53 MB', 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:33', '2024-11-13 11:30:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (10, 0, '图像去雾', 'D4', 'D4/weights_reconstruct.pth', '88.16 MB', 'algorithm.D4.run', '', 1,
        '2024-11-13 12:39:45', '2024-11-13 12:39:45', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (11, 0, '图像去雾', 'DaclipUir', 'daclip-uir/daclip_ViT-B-32.pt', '1.62 GB', 'algorithm.DaclipUir.run',
        'DaclipUir 是一种先进的图像去雾模型，它结合了深度学习与物理模型，通过优化图像的对比度和色彩，有效去除雾霾，提高图像的清晰度。该模型特别注重保留图像的细节和自然度，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:39:54', '2024-11-13 12:39:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (12, 0, '图像去雾', 'DCPDN', 'DCPDN/netG_epoch_8.pth', '255.55 MB', 'algorithm.DCPDN.run',
        'DCPDN 是一种基于深度学习的图像去雾方法，通过大气散射模型和密集连接的编码器-解码器结构，估计透射率图并进行去雾。该模型利用多级金字塔池化模块，提高了透射率估计的准确性，从而改善了去雾效果',
        1, '2024-11-13 12:40:03', '2024-11-13 12:40:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (13, 0, '图像去雾', 'DCP', '/', null, 'algorithm.DCP.run',
        'DCP 是由何凯明等人在2009年提出的经典去雾算法，基于暗原色先验理论。该算法假设无雾图像的局部区域中至少有一个颜色通道的亮度值非常低。通过估计大气光和透射率，DCP 能够有效地去除图像中的雾霾，恢复图像的清晰度',
        1, '2024-11-13 12:40:13', '2024-11-13 12:40:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (14, 0, '图像去雾', 'DEANet', 'DEA-Net', null, 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:40:20', '2024-11-13 12:40:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (15, 14, '图像去雾', 'HAZE4k模型', 'DEA-Net/HAZE4K/PSNR3426_SSIM9885.pth', '14 MB', 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:40:55', '2024-11-13 12:40:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (16, 14, '图像去雾', 'ITS模型', 'DEA-Net/ITS/PSNR4131_SSIM9945.pth', '14 MB', 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:41:02', '2024-11-13 12:41:02', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (17, 14, '图像去雾', 'OTS模型', 'DEA-Net/OTS/PSNR3659_SSIM9897.pth', '14 MB', 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:41:09', '2024-11-13 12:41:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (18, 0, '图像去雾', 'Dehamer', 'Dehamer', null, 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:41:26', '2024-11-13 12:41:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (19, 18, '图像去雾', 'dense-haze模型', 'Dehamer/dense/PSNR1662_SSIM05602.pt', '511.68 MB',
        'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:41:56', '2024-11-13 12:41:56', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (20, 18, '图像去雾', 'indoor', 'Dehamer/indoor/PSNR3663_ssim09881.pt', '511.68 MB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:02', '2024-11-13 12:42:02', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (21, 18, '图像去雾', 'NH-HAZE-模型', 'Dehamer/NH/PSNR2066_SSIM06844.pt', '511.68 MB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:10', '2024-11-13 12:42:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (22, 18, '图像去雾', 'outdoor', 'Dehamer/outdoor/PSNR3518_SSIM09860.pt', '511.68 MB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:17', '2024-11-13 12:42:17', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (23, 0, '图像去雾', 'DehazeFormer', 'DehazeFormer', null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:42:35', '2024-11-13 12:42:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (24, 23, '图像去雾', 'indoor', 'DehazeFormer/indoor', null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:42:57', '2024-11-13 12:42:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (25, 24, '图像去雾', 'indoor-b', 'DehazeFormer/indoor/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:45:52', '2024-11-13 12:45:52', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (26, 24, '图像去雾', 'indoor-d', 'DehazeFormer/indoor/dehazeformer-d.pth', '21.22 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:45:59', '2024-11-13 12:45:59', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (27, 24, '图像去雾', 'indoor-l', 'DehazeFormer/indoor/dehazeformer-l.pth', '98.22 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:05', '2024-11-13 12:46:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (28, 24, '图像去雾', 'indoor-m', 'DehazeFormer/indoor/dehazeformer-m.pth', '18.51 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:12', '2024-11-13 12:46:12', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (29, 24, '图像去雾', 'indoor-s', 'DehazeFormer/indoor/dehazeformer-s.pth', '5.46 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:19', '2024-11-13 12:46:19', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (30, 24, '图像去雾', 'indoor-t', 'DehazeFormer/indoor/dehazeformer-t.pth', '2.9 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:25', '2024-11-13 12:46:25', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (31, 24, '图像去雾', 'indoor-w', 'DehazeFormer/indoor/dehazeformer-w.pth', '38.06 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:33', '2024-11-13 12:46:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (32, 23, '图像去雾', 'outdoor', 'DehazeFormer/outdoor', null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:44', '2024-11-13 12:46:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (33, 32, '图像去雾', 'outdoor-b', 'DehazeFormer/outdoor/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:32', '2024-11-13 12:51:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (34, 32, '图像去雾', 'outdoor-m', 'DehazeFormer/outdoor/dehazeformer-m.pth', '18.51 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:38', '2024-11-13 12:51:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (35, 32, '图像去雾', 'outdoor-s', 'DehazeFormer/outdoor/dehazeformer-s.pth', '5.46 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:44', '2024-11-13 12:51:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (36, 32, '图像去雾', 'outdoor-t', 'DehazeFormer/outdoor/dehazeformer-t.pth', '2.9 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:50', '2024-11-13 12:51:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (37, 23, '图像去雾', 'reside6k', 'DehazeFormer/reside6k', null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:00', '2024-11-13 12:52:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (38, 37, '图像去雾', 'reside6k-b', 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:13', '2024-11-13 12:52:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (39, 37, '图像去雾', 'reside6k-b', 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:19', '2024-11-13 12:52:19', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (40, 37, '图像去雾', 'reside6k-b', 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:26', '2024-11-13 12:52:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (41, 37, '图像去雾', 'reside6k-b', 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB',
        'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:34', '2024-11-13 12:52:34', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (42, 23, '图像去雾', 'rshaze', 'DehazeFormer/rshaze', null, 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 12:52:50', '2024-11-13 12:52:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (43, 42, '图像去雾', 'rshaze-b', 'DehazeFormer/rshaze/dehazeformer-b.pth', '10.71 MB', 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 12:59:58', '2024-11-13 12:59:58', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (44, 42, '图像去雾', 'rshaze-m', 'DehazeFormer/rshaze/dehazeformer-m.pth', '18.51 MB', 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:05', '2024-11-13 13:00:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (45, 42, '图像去雾', 'rshaze-s', 'DehazeFormer/rshaze/dehazeformer-s.pth', '5.46 MB', 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:18', '2024-11-13 13:00:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (46, 42, '图像去雾', 'rshaze-t', 'DehazeFormer/rshaze/dehazeformer-t.pth', '2.9 MB', 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:25', '2024-11-13 13:00:25', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (47, 0, '图像去雾', 'FCD', 'FCD/framework_da_230221_121802_gen.pth', '592.75 MB', 'algorithm.FCD.run',
        'FCD 是一种基于全卷积网络的图像去雾方法，通过密集连接的卷积层进行端到端的去雾处理。该模型简化了模型结构，提高了计算效率，适用于实时去雾应用',
        1, '2024-11-13 13:00:33', '2024-11-13 13:00:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (48, 0, '图像去雾', 'FFANet', 'FFA-Net', null, 'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:00:40', '2024-11-13 13:00:40', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (49, 48, '图像去雾', 'its', 'FFA-Net/its_train_ffa_3_19.pk', '21.26 MB', 'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:12:54', '2024-11-13 13:12:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (50, 48, '图像去雾', 'ots', 'FFA-Net/ots_train_ffa_3_19.pk', '25.39 MB', 'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:13:01', '2024-11-13 13:13:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (51, 0, '图像去雾', 'FogRemoval', 'FogRemoval/NH-HAZE_params_0100000.pt', '512.01 MB',
        'algorithm.FogRemoval.run',
        'FogRemoval 是一种多阶段的图像去雾方法，通过逐步优化图像质量，实现自然的去雾效果。该模型结合了物理模型和深度学习，能够在不同光照条件下有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:13:11', '2024-11-13 13:13:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (52, 0, '图像去雾', 'GCANet', 'GCANet/wacv_gcanet_dehaze.pth', '2.69 MB', 'algorithm.GCANet.run',
        'GCANet 是一种利用全局上下文模块的图像去雾模型，通过增强模型对全局信息的理解，改善去雾结果。该模型在处理复杂场景时，能够有效保留图像的结构和细节，提高去雾效果',
        1, '2024-11-13 13:13:18', '2024-11-13 13:13:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (53, 0, '图像去雾', 'GridDehazeNet', 'GridDehazeNet', null, 'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:27', '2024-11-13 13:13:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (54, 53, '图像去雾', 'indoor', 'GridDehazeNet/indoor_haze_best_3_6', '3.71 MB', 'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:38', '2024-11-13 13:13:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (55, 53, '图像去雾', 'outdoor', 'GridDehazeNet/outdoor_haze_best_3_6', '3.71 MB', 'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:44', '2024-11-13 13:13:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (56, 0, '图像去雾', 'ImgRestorationSde', 'image-restoration-sde', null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:13:52', '2024-11-13 13:13:52', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (57, 56, '图像去模糊', 'deblurring', 'image-restoration-sde/deblurring/ir-sde-deblurring.pth', '523.23 MB',
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:32', '2024-11-13 13:22:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (58, 56, '图像去噪', 'denoising', 'image-restoration-sde/denoising/ir-sde-denoising.pth', '523.19 MB',
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:39', '2024-11-13 13:22:39', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (59, 56, '图像去雨', 'deraining', 'image-restoration-sde/deraining', null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:46', '2024-11-13 13:22:46', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (60, 59, '图像去雨', 'deraining-H100', 'image-restoration-sde/deraining/ir-sde-derainH100.pth', '523.23 MB',
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:55', '2024-11-13 13:22:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (61, 59, '图像去雨', 'deraining-L100', 'image-restoration-sde/deraining/ir-sde-derainL100.pth', '523.23 MB',
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:23:01', '2024-11-13 13:23:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (62, 0, '图像去雾', 'ITBDehaze', 'ITBdehaze/best.pkl', '423.84 MB', 'algorithm.ITBDehaze.run',
        'ITBDehaze (Image Texture and Boundary Dehazing)是一种利用图像的纹理和边界信息的图像去雾模型，通过多尺度处理增强去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度24',
        1, '2024-11-13 13:23:09', '2024-11-13 13:23:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (63, 0, '图像去雾', 'LightDehazeNet', 'LightDehazeNet/trained_LDNet.pth', '122.61 KB',
        'algorithm.LightDehazeNet.run',
        'LightDehazeNet 是一种轻量级的图像去雾模型，适用于移动设备上的实时去雾应用。该模型通过优化网络结构，减少了计算复杂度，同时保持了较高的去雾效果',
        1, '2024-11-13 13:23:15', '2024-11-13 13:23:15', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (64, 0, '图像去雾', 'LKDNet', 'LKDNet', null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:22', '2024-11-13 13:23:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (65, 64, '图像去雾', 'ITS', 'LKDNet/ITS', null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:29', '2024-11-13 13:23:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (66, 65, '图像去雾', 'ITS-b', 'LKDNet/ITS/LKD-b/LKD-b.pth', '4.94 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:47', '2024-11-13 13:23:47', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (67, 65, '图像去雾', 'ITS-l', 'LKDNet/ITS/LKD-l/LKD-l.pth', '9.67 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:53', '2024-11-13 13:23:53', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (68, 65, '图像去雾', 'ITS-s', 'LKDNet/ITS/LKD-s/LKD-s.pth', '2.57 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:58', '2024-11-13 13:23:58', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (69, 65, '图像去雾', 'ITS-t', 'LKDNet/ITS/LKD-t/LKD-t.pth', '1.39 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:08', '2024-11-13 13:24:08', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (70, 64, '图像去雾', 'OTS', 'LKDNet/OTS', null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:15', '2024-11-13 13:24:15', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (71, 70, '图像去雾', 'OTS-b', 'LKDNet/OTS/LKD-b/LKD-b.pth', '4.94 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:21', '2024-11-13 13:24:21', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (72, 70, '图像去雾', 'OTS-l', 'LKDNet/OTS/LKD-l/LKD-l.pth', '9.67 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:27', '2024-11-13 13:24:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (73, 70, '图像去雾', 'OTS-s', 'LKDNet/OTS/LKD-s/LKD-s.pth', '2.57 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:35', '2024-11-13 13:24:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (74, 70, '图像去雾', 'OTS-t', 'LKDNet/OTS/LKD-t/LKD-t.pth', '1.39 MB', 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:41', '2024-11-13 13:24:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (75, 0, '图像去雾', 'MADN', 'MADN/model.pth', '2.13 MB', 'algorithm.MADN.run',
        'MADN (Multi-Adversarial Domain Network): MADN 是一种基于多对抗域网络的图像去雾模型，通过域适应技术，提高了模型对不同场景的适应能力。该模型在处理真实世界中的雾霾图像时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:24:50', '2024-11-13 13:24:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (76, 0, '图像去雾', 'MB-TaylorFormer', 'MB-TaylorFormer', null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:24:57', '2024-11-13 13:24:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (77, 76, '图像去雾', 'dense-haze', 'MB-TaylorFormer', null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:22', '2024-11-13 13:28:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (78, 77, '图像去雾', 'dense-haze-b', 'MB-TaylorFormer/densehaze-MB-TaylorFormer-B.pth', '10.49 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:29', '2024-11-13 13:28:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (79, 77, '图像去雾', 'dense-haze-l', 'MB-TaylorFormer/densehaze-MB-TaylorFormer-L.pth', '29.04 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:36', '2024-11-13 13:28:36', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (80, 76, '图像去雾', 'its', 'MB-TaylorFormer/ITS-MB-TaylorFormer-L.pth', '29.04 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:43', '2024-11-13 13:28:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (81, 76, '图像去雾', 'ohaze', 'MB-TaylorFormer/ohaze-MB-TaylorFormer-B.pth', '10.49 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:49', '2024-11-13 13:28:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (82, 76, '图像去雾', 'ots', 'MB-TaylorFormer', null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:00', '2024-11-13 13:29:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (83, 82, '图像去雾', 'ots-b', 'MB-TaylorFormer/OTS-MB-TaylorFormer-B.pth', '10.51 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:06', '2024-11-13 13:29:06', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (84, 82, '图像去雾', 'ots-l', 'MB-TaylorFormer/OTS-MB-TaylorFormer-L.pth', '29.04 MB',
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:12', '2024-11-13 13:29:12', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (85, 0, '图像去雾', 'MixDehazeNet', 'MixDehazeNet', null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 13:29:57', '2024-11-13 13:29:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (86, 85, '图像去雾', 'haze4k', 'MixDehazeNet/haze4k', null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:20', '2024-11-13 14:26:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (87, 86, '图像去雾', 'haze4k-l', 'MixDehazeNet/haze4k/MixDehazeNet-l.pth', '143.93 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:47', '2024-11-13 14:26:47', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (88, 85, '图像去雾', 'indoor', 'MixDehazeNet/indoor', null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:54', '2024-11-13 14:26:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (89, 88, '图像去雾', 'indoor-b', 'MixDehazeNet/indoor/MixDehazeNet-b.pth', '72.44 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:00', '2024-11-13 14:27:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (90, 88, '图像去雾', 'indoor-l', 'MixDehazeNet/indoor/MixDehazeNet-l.pth', '143.93 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:06', '2024-11-13 14:27:06', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (91, 85, '图像去雾', 'outdoor', 'MixDehazeNet/outdoor', null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:13', '2024-11-13 14:27:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (92, 91, '图像去雾', 'outdoor-b', 'MixDehazeNet/outdoor/MixDehazeNet-b.pth', '72.44 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:26', '2024-11-13 14:27:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (93, 91, '图像去雾', 'outdoor-b', 'MixDehazeNet/outdoor/MixDehazeNet-l.pth', '143.93 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:35', '2024-11-13 14:27:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (94, 91, '图像去雾', 'outdoor-b', 'MixDehazeNet/outdoor/MixDehazeNet-s.pth', '36.69 MB',
        'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:41', '2024-11-13 14:27:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (95, 0, '图像去雾', 'MSFNet', 'MSFNet', null, 'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:49', '2024-11-13 14:27:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (96, 95, '图像去雾', 'indoor', 'MSFNet/indoor.pth', '4.01 MB', 'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:58', '2024-11-13 14:27:58', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (97, 95, '图像去雾', 'outdoor', 'MSFNet/outdoor.pth', '3.98 MB', 'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:04', '2024-11-13 14:28:04', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (98, 0, '图像去雾', 'PSD', 'PSD', null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:10', '2024-11-13 14:28:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (99, 98, '图像去雾', 'PSD-MSBDN', 'PSD/PSB-MSBDN', '126.4 MB', 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:19', '2024-11-13 14:28:19', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (100, 98, '图像去雾', 'PSD-FFANET', 'PSD/PSD-FFANET', '23.84 MB', 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:33', '2024-11-13 14:28:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (101, 98, '图像去雾', 'PSD-GCANET', 'PSD/PSD-GCANET', '9.23 MB', 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:42', '2024-11-13 14:28:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (102, 0, '图像去雾', 'RIDCP', 'RIDCP/pretrained_RIDCP.pth', '116.41 MB', 'algorithm.RIDCP.run',
        '目前图像去雾领域缺乏强大的先验知识，作者提出在 VQGAN1使用大规模高质量数据集，预训练出一个离散码本，封装高质量先验（HQPs）；并且引入了一种提取特征能力较强的编码器 E，以及设计了一个具有归一化特征对齐模块（NFA）的解码器 G ，共同构建出基于高质量码本先验的真实图像去雾网络（RIDCP）',
        1, '2024-11-13 14:28:49', '2024-11-13 14:28:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (103, 0, '图像去雾', 'SCANet', 'SCANet', null, 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:56', '2024-11-13 14:28:56', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (104, 103, '图像去雾', 'SCANet-40', 'SCANet/Gmodel_40.tar', '27.7 MB', 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:05', '2024-11-13 14:29:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (105, 103, '图像去雾', 'SCANet-105', 'SCANet/Gmodel_105.tar', '27.7 MB', 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:11', '2024-11-13 14:29:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (106, 103, '图像去雾', 'SCANet-120', 'SCANet/Gmodel_120.tar', '27.68 MB', 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:16', '2024-11-13 14:29:16', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (107, 0, '图像去雾', 'SGIDPFF', 'SGID-PFF', null, 'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:22', '2024-11-13 14:29:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (108, 107, '图像去雾', 'indoor', 'SGID-PFF/SOTS_indoor.pt', '52.94 MB', 'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:28', '2024-11-13 14:29:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (109, 107, '图像去雾', 'SGIDPFF', 'SGID-PFF/SOTS_outdoor.pt', '52.94 MB', 'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:33', '2024-11-13 14:29:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (110, 0, '图像去雾', 'TSDNet', 'TSDNet/GNet.tar', '13.94 MB', 'algorithm.TSDNet.run',
        'TSDNet (Temporal-Spatial-Depth Network): TSDNet 是一种时间-空间-深度联合建模的图像去雾模型，通过优化视频序列的去雾效果，提高视频去雾的连贯性和自然度。该模型在处理视频序列时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的视频去雾任务。',
        1, '2024-11-13 14:29:43', '2024-11-13 14:29:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (111, 0, '图像去雾', 'WPXNet', 'WPXNet', null, 'algorithm.WPXNet.run',
        '引入无雾图像训练得到离散码本，封装具有原有图像色彩和结构的高质量先验知识。随后构建一种双分支神经网络结构，即先验匹配分支和通道注意力分支，利用邻域注意力和通道注意力提取有雾图像全局特征并学习浓雾区域与底层场景之间复杂交互特征，通过特征融合模块对两个分支提取的特征进行融合。将高质量先验约束码本与有雾图像特征通过一种可控距离重计算操作进行匹配，从而替换图像中受到雾影响的区域。本发明对原有雾图像进行重建实现了端到端的图像去雾流程，在保留原图像细节和纹理结构的情况下，提高了有雾图像的清晰度和可识别度。',
        1, '2024-11-28 14:03:07', '2024-11-28 14:03:07', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (112, 111, '图像去雾', 'DENSE-HAZE', 'WPXNet/dense-haze.pth', '151.75 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:05', '2024-11-28 14:04:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (113, 111, '图像去雾', 'I-HAZE', 'WPXNet/i-haze.pth', '151.75 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:24', '2024-11-28 14:04:24', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (114, 111, '图像去雾', 'O-HAZE', 'WPXNet/o-haze.pth', '151.74 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:45', '2024-11-28 14:04:45', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (115, 111, '图像去雾', 'NH-HAZE-20', 'WPXNet/nh-haze-20.pth', '151.74 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:06', '2024-11-28 14:05:06', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (116, 111, '图像去雾', 'NH-HAZE-21', 'WPXNet/nh-haze-21.pth', '151.74 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:16', '2024-11-28 14:05:16', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, path, size, import_path, description, status, create_time,
                           update_time, create_by, update_by)
values (117, 111, '图像去雾', 'NH-HAZE-23', 'WPXNet/nh-haze-23.pth', '151.75 MB', 'algorithm.WPXNet.run',
        '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:23', '2024-11-28 14:05:23', 2, 2);

DROP TABLE IF EXISTS `sys_dataset_file`;
CREATE TABLE `sys_dataset_file`
(
    `id`            bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `dataset_id`    bigint      NOT NULL COMMENT '所属数据集id',
    `image_item_id` bigint      NOT NULL COMMENT '所属数据项id',
    `file_id`       bigint      NOT NULL COMMENT '文件id',
    `type`          varchar(64) NOT NULL COMMENT '图片类型（清晰图、雾霾图、分割图等）',
    `thumbnail`     boolean     NOT NULL DEFAULT FALSE COMMENT '是否为缩略图',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据集图片关联表';

DROP TABLE IF EXISTS `sys_wpx_file`;
CREATE TABLE `sys_wpx_file`
(
    `id`             bigint          NOT NULL AUTO_INCREMENT COMMENT 'id',
    `origin_file_id` bigint COMMENT '旧文件id',
    `origin_md5`     char(32) unique NOT NULL COMMENT '旧文件的MD5值',
    `origin_path`    varchar(255)    NOT NULL COMMENT '旧文件路径',
    `new_file_id`    bigint COMMENT '新文件id',
    `new_path`       varchar(255)    NOT NULL COMMENT '新文件路径',
    `new_md5`        char(32) unique NOT NULL COMMENT '新文件的MD5值',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='WPX文件表';

# I-HAZE
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/01_indoor_GT.jpg', '3e096042c85dfa8043e1ad13e171009f', 'WPX/I-HAZE/clean/01_GT.png',
        'a3969e85941f9ecb04ec490af673023b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/02_indoor_GT.jpg', 'ab59c8cbb6978d5ea109c421dfe877ce', 'WPX/I-HAZE/clean/02_GT.png',
        '63c26a80da3b5acd0f87ccec2f734475');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/03_indoor_GT.jpg', 'd94a926bd522be17224c371b18acdbe8', 'WPX/I-HAZE/clean/03_GT.png',
        '56fee33d3d44fca00c95280c065001fe');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/04_indoor_GT.jpg', '8dbf776b6f6ad61f4bcd7d0e626ddb28', 'WPX/I-HAZE/clean/04_GT.png',
        '5830f7ec96e1423857424535430244ad');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/05_indoor_GT.jpg', '4e45a3b29df3f304cb7d0e2fdab6ac79', 'WPX/I-HAZE/clean/05_GT.png',
        'aca3c8b359a42da89903149e73b317e4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/06_indoor_GT.jpg', 'e3984aad71b1b74a491ef782a33adb3e', 'WPX/I-HAZE/clean/06_GT.png',
        'f7d0f2faade54a4715492fc41f52d24f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/07_indoor_GT.jpg', '5853a4fc877c494ecac3b68bfe07b7fe', 'WPX/I-HAZE/clean/07_GT.png',
        '3d8539780656c6a9d63fa26a3500ddff');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/08_indoor_GT.jpg', 'd7ccb0067fcabb008c757a677529c8fa', 'WPX/I-HAZE/clean/08_GT.png',
        '2a7061648989771f8a82a967109f2e87');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/09_indoor_GT.jpg', 'de1e0e65175186cd8d3c98b09fb77136', 'WPX/I-HAZE/clean/09_GT.png',
        '281c623af331f59bb243991ee99979a3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/10_indoor_GT.jpg', '6f482676e47cc7371302cd70f66d3a5b', 'WPX/I-HAZE/clean/10_GT.png',
        'dd9606d74b7bf45dcf63e94d5f7e11ca');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/11_indoor_GT.jpg', 'e227e6dd8392e5ea9555a4c5c043e44e', 'WPX/I-HAZE/clean/11_GT.png',
        'b760b34bf27de20127fa5554adc36cee');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/12_indoor_GT.jpg', '8b74e81df8016d4fada6d048d81aa741', 'WPX/I-HAZE/clean/12_GT.png',
        'e494ad259a98c1d51289723f13772deb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/13_indoor_GT.jpg', 'f31bfa3f59d50f6ad8bc0861997d884b', 'WPX/I-HAZE/clean/13_GT.png',
        '4979c014351e3c9a5df270c6191f4857');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/14_indoor_GT.jpg', '74be7e2a5bd9f8815a514b49ac090c7d', 'WPX/I-HAZE/clean/14_GT.png',
        '5e90997cb2960bf5b302f940a8378271');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/15_indoor_GT.jpg', '7e37cfb1d826a32b5b4910360748e2e3', 'WPX/I-HAZE/clean/15_GT.png',
        '3e09513436b969e31113c8d15556b7ba');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/16_indoor_GT.jpg', '6e9ac3aaeb180c0aae3874741c38496b', 'WPX/I-HAZE/clean/16_GT.png',
        '7e8f970dd18d64a6086fe9e554183456');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/17_indoor_GT.jpg', '90cb711cd4c4ef6fb7ab08e2398ec273', 'WPX/I-HAZE/clean/17_GT.png',
        '9756052ef53d5d28e8e18289e9283a50');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/18_indoor_GT.jpg', '6b365a85a25f91b0297922a1917e2919', 'WPX/I-HAZE/clean/18_GT.png',
        'ead504caf8e1921cd82385b95cfbc927');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/19_indoor_GT.jpg', 'cc35d01dad7ffe90b7a8678acbd39cbc', 'WPX/I-HAZE/clean/19_GT.png',
        'd19887a87368502e3573a6b6b6e36a60');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/20_indoor_GT.jpg', 'fb7932a300a420011c4bac96f72ae969', 'WPX/I-HAZE/clean/20_GT.png',
        '5cbf82d53f7e2c4509b7bdfec482127f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/21_indoor_GT.jpg', 'ac4c69779470f594bb8a28a18ad4dc65', 'WPX/I-HAZE/clean/21_GT.png',
        '501e01fbaa2afcfe75bec4e68c8aa500');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/22_indoor_GT.jpg', 'bdffa2cd82bfc50558c763fb81ed999e', 'WPX/I-HAZE/clean/22_GT.png',
        'e46dbb4b825b187c9b686cc3779f9290');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/23_indoor_GT.jpg', '40e425d4b81629e15fcc4ae770c4f352', 'WPX/I-HAZE/clean/23_GT.png',
        'dd3eaa9d9de64b0ff5510d1b4c2319b9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/24_indoor_GT.jpg', '97d6ce6561872a02a09b3c3f7a172b12', 'WPX/I-HAZE/clean/24_GT.png',
        'e719eafd33425e42469349349c3badc1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/25_indoor_GT.jpg', 'e03f265097c1ac7a62c7ad2ef92e0ad1', 'WPX/I-HAZE/clean/25_GT.png',
        'a456ec8f0ed2f5c83e7b2aa20c9cf7b1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/31_indoor_GT.jpg', '205cce1f5614c5f136bdd665c693aef3', 'WPX/I-HAZE/clean/26_GT.png',
        'fc79cd5d24f216951852bf7a6d8fd5f0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/32_indoor_GT.jpg', 'f5822eb8a9a9aa211003313ac4e5d7c4', 'WPX/I-HAZE/clean/27_GT.png',
        'c2138e6b91a06963e16184ce82876f74');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/33_indoor_GT.jpg', '06acd13cbd727e12f4b24a09a5c535c1', 'WPX/I-HAZE/clean/28_GT.png',
        'c661ee9d63a0e9299c30bc3431130601');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/34_indoor_GT.jpg', '6af927ecea00038801da44bde222674b', 'WPX/I-HAZE/clean/29_GT.png',
        'eee691a69ec8f98bc2b2c47edc1e6e88');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/clean/35_indoor_GT.jpg', 'd1accd5fd652198e9e310584e1b3290e', 'WPX/I-HAZE/clean/30_GT.png',
        '580332a5471cb239f673a14f9c4c5dae');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/01_indoor_hazy.jpg', '2462f35fa6d3ce3c9c7b567d84817e36', 'WPX/I-HAZE/hazy/01_hazy.png',
        'a7a90d6e9673dcd997e948ac4cf8c9fa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/02_indoor_hazy.jpg', 'af3245cdecdd6dea9879ada4ce2aab32', 'WPX/I-HAZE/hazy/02_hazy.png',
        '988dc9b884641363a4ff78dbea8f0024');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/03_indoor_hazy.jpg', '05ac82a2b6f48df6aa5bf1eecd480ac2', 'WPX/I-HAZE/hazy/03_hazy.png',
        '83bb848dda51a0d79865354d788706d8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/04_indoor_hazy.jpg', '235f360ba476801d0280c859de98cd2b', 'WPX/I-HAZE/hazy/04_hazy.png',
        '071f72021acbf8c2dd958361ae5ecd91');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/05_indoor_hazy.jpg', '38766fda2013a3af2d3f8ab723b10f69', 'WPX/I-HAZE/hazy/05_hazy.png',
        '4a70d40c59e9c570b4a1c7cbdb60ee52');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/06_indoor_hazy.jpg', 'b14c6061c44b2421703fd9caa02d0fcf', 'WPX/I-HAZE/hazy/06_hazy.png',
        'a809b64c79ed9cd50909605b0f8c37ae');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/07_indoor_hazy.jpg', '33a923755280f39bb0c98ae7aa0a5b87', 'WPX/I-HAZE/hazy/07_hazy.png',
        '7ce24eb9bed98fa4f65640245aeda168');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/08_indoor_hazy.jpg', 'cda5c91d25bd7056dee7672304856ee8', 'WPX/I-HAZE/hazy/08_hazy.png',
        'c9a4a49e69863f3913bc124a6faf1709');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/09_indoor_hazy.jpg', '5ff8e242e3f07af584f499e7298f4b9e', 'WPX/I-HAZE/hazy/09_hazy.png',
        '5f0b69cf88477cc5c7dc4bd019c90112');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/10_indoor_hazy.jpg', 'c59d12b96af930f475b23ba2776a69e7', 'WPX/I-HAZE/hazy/10_hazy.png',
        'e7edbcd5c3e4d8c7173ae59802d81344');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/11_indoor_hazy.jpg', 'ee4179b7ca1f3ef2d4decdaf42037cef', 'WPX/I-HAZE/hazy/11_hazy.png',
        '2927a1a025fdc3d6c2a5abbd014581f4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/12_indoor_hazy.jpg', 'a0cf48d45cb13220b5174ff27fe25c55', 'WPX/I-HAZE/hazy/12_hazy.png',
        'd1a7b7b8f5eda40cf0012270b541b142');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/13_indoor_hazy.jpg', '0d1fa828c989169a46555a71d38187e0', 'WPX/I-HAZE/hazy/13_hazy.png',
        '2312f41b90b22d9a047d34b567d03d45');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/14_indoor_hazy.jpg', '81d48ba922ec01f5bd00d63839a54071', 'WPX/I-HAZE/hazy/14_hazy.png',
        '8fb4d64c33ab663c725e07af24baa4e4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/15_indoor_hazy.jpg', '1765b090c72d6a08f1462ade5cc135d6', 'WPX/I-HAZE/hazy/15_hazy.png',
        '0cb508f7a63039a30f12c35df1949865');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/16_indoor_hazy.jpg', 'c49d583541392ade03ec90978d36b70d', 'WPX/I-HAZE/hazy/16_hazy.png',
        '37c076f52077f18511cd776cb5c706cd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/17_indoor_hazy.jpg', '9378469982f70e294eb7142745f69db2', 'WPX/I-HAZE/hazy/17_hazy.png',
        'ac88122e35fa778677e0a9aecd6ac3ef');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/18_indoor_hazy.jpg', '30f2587a7e9e1a761dda5200340433c3', 'WPX/I-HAZE/hazy/18_hazy.png',
        'bc780db4e588748a4df78fc797c420cc');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/19_indoor_hazy.jpg', 'af411605f25b3bde28ff9f26dfc517a7', 'WPX/I-HAZE/hazy/19_hazy.png',
        '73e783528db61eef980c18e332ac528a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/20_indoor_hazy.jpg', 'efa9e7211ef2b5587b313eb93265917b', 'WPX/I-HAZE/hazy/20_hazy.png',
        '0f23bc48e0760a11f1d6764f46b5dee4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/21_indoor_hazy.jpg', 'f03d746a68741969e5cbbcdfe6c1dbae', 'WPX/I-HAZE/hazy/21_hazy.png',
        '34f3ed7e094fef941210deee4a0e0c0f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/22_indoor_hazy.jpg', '47c904291722e690368f294937daa60f', 'WPX/I-HAZE/hazy/22_hazy.png',
        'fdb75e46e4da9a3c1f61ca3d01a5a8a6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/23_indoor_hazy.jpg', 'da64f861b3b2ebd0a9f32d13ff9568d5', 'WPX/I-HAZE/hazy/23_hazy.png',
        'b055994f8315837e34529b94a9d27a05');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/24_indoor_hazy.jpg', 'c6f95dac8263069146146af912d1dd2a', 'WPX/I-HAZE/hazy/24_hazy.png',
        '0cf7a68ffc01d5a61e0a0b53b40a9139');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/25_indoor_hazy.jpg', 'b688eb5dba985e28bff3e06c4097b54e', 'WPX/I-HAZE/hazy/25_hazy.png',
        '14bbc2293dfc2ae5734a742e760ea73e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/31_indoor_hazy.jpg', 'b2b92799bf6632f6d0492c0312379ddf', 'WPX/I-HAZE/hazy/26_hazy.png',
        '82a89c83ffb4439b3e046c85f168d8b3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/32_indoor_hazy.jpg', '9d71aa70e5330be1c4b0e75f5313410f', 'WPX/I-HAZE/hazy/27_hazy.png',
        '6fa048eee7007972e8a00a35eaa118fe');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/33_indoor_hazy.jpg', '60b8e4ac00adcc32fc1066bb95bf505f', 'WPX/I-HAZE/hazy/28_hazy.png',
        'a00d766686a4a9c8822db176767cf5c6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/34_indoor_hazy.jpg', '4b58e0a14f0a8b8cd8bca94060aca697', 'WPX/I-HAZE/hazy/29_hazy.png',
        'ee97fa385d13595acfcbe31b933e0518');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('I-HAZE/hazy/35_indoor_hazy.jpg', 'feae1f3d32624f4b1b9afa0e226b6653', 'WPX/I-HAZE/hazy/30_hazy.png',
        '1a772d4d480b1f778ae726c089103abd');
# O-HAZE
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/01_outdoor_GT.jpg', '8d05273c843541afb16d79d455ea027f', 'WPX/O-HAZE/clean/01_GT.png',
        '565c475dd92509c351c259f26ae4f185');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/02_outdoor_GT.jpg', '0bb5257e8932a76f2141abb31f0c1dcd', 'WPX/O-HAZE/clean/02_GT.png',
        '5e8cfd50ff1c6700848b1e210d65b3d8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/03_outdoor_GT.JPG', '914f2c70da0af191deed85d3afca8d9f', 'WPX/O-HAZE/clean/03_GT.png',
        'cddb76ba38874ef2c4130a73c7c2cd70');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/04_outdoor_GT.jpg', 'feec57fe1040c7fe26849bcc39b0c173', 'WPX/O-HAZE/clean/04_GT.png',
        '72bc518be2be3d526dfe17d38ba0b020');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/05_outdoor_GT.jpg', 'ab38509178716f1e48bff81df63aa093', 'WPX/O-HAZE/clean/05_GT.png',
        'a5d829b67025acfa8c42118ba6c9efdf');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/06_outdoor_GT.jpg', 'de1bcb37c9a7f503211b5cef7162b539', 'WPX/O-HAZE/clean/06_GT.png',
        '92fd74b7437678a8491dfc4786072643');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/07_outdoor_GT.jpg', '6ec100a66571279c4e594d979ce927a2', 'WPX/O-HAZE/clean/07_GT.png',
        'bafe00e4f1d1b27bbc36c305f6857b3c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/08_outdoor_GT.jpg', '475e7c0f8aff7651c6111993e53eac20', 'WPX/O-HAZE/clean/08_GT.png',
        'cd773fb4999fb8b9e68333b93bbcde13');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/09_outdoor_GT.jpg', '0a135043aa183efd927ff4bd14e98f7f', 'WPX/O-HAZE/clean/09_GT.png',
        '0c258095625ce184cb2c291b33b4fe86');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/10_outdoor_GT.jpg', 'fbc99e110e74c5160dba810c48be3469', 'WPX/O-HAZE/clean/10_GT.png',
        '195e9fb26d824302ce7b5107acbb5ca9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/11_outdoor_GT.jpg', '28cda358fb63cf4942840ca6fca93794', 'WPX/O-HAZE/clean/11_GT.png',
        'b054ae383cbd039449b0fe48db964ccb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/12_outdoor_GT.jpg', '7c56802c708e8025968cc1d756ec6d27', 'WPX/O-HAZE/clean/12_GT.png',
        'b934da3fae4112192fdd19ff665a7b8c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/13_outdoor_GT.jpg', '08f098da10eb1dd29abe3bea7d8bafa7', 'WPX/O-HAZE/clean/13_GT.png',
        '751f7cbff907fbede30b7459572e3f3a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/14_outdoor_GT.jpg', '4651dd896506c1f380932a2b80438e95', 'WPX/O-HAZE/clean/14_GT.png',
        '3f0f942672ef7adc0722ff1a5ee8a6f3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/15_outdoor_GT.jpg', '2d7d39484a567f69fbf78a621bfeab20', 'WPX/O-HAZE/clean/15_GT.png',
        'f65ec97a52bba8cc4da937997a45966a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/16_outdoor_GT.jpg', '5f78943169bf8cadf7cbe37d59857874', 'WPX/O-HAZE/clean/16_GT.png',
        '644950ae103216363b691156794c7a69');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/17_outdoor_GT.jpg', '4c7260e5854271be85a52d8c0e866ed4', 'WPX/O-HAZE/clean/17_GT.png',
        'cd878b2455a5f8dc2aeb0565639029de');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/18_outdoor_GT.jpg', 'ebab052909ef1fb49ac6e5a26e28a40d', 'WPX/O-HAZE/clean/18_GT.png',
        'e61a9fd2005758a80cdf7aeb663af1e8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/19_outdoor_GT.jpg', '08950f10df2f53bcca63a6808499065a', 'WPX/O-HAZE/clean/19_GT.png',
        'f7ff918cbf35acca7719173c0e6dabb8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/20_outdoor_GT.jpg', 'e95f17d2a0e0db5f773df173a3ea9a69', 'WPX/O-HAZE/clean/20_GT.png',
        '2b628b6d38ed59b079cc2998cde84450');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/21_outdoor_GT.JPG', 'de27024bd8e58846130ecb8352aae8e0', 'WPX/O-HAZE/clean/21_GT.png',
        'eb322d08f96c23a3a04c943689c12db6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/22_outdoor_GT.jpg', '6651ec7e827e88e716414eaf981d4375', 'WPX/O-HAZE/clean/22_GT.png',
        '3aca58afd12ef12cc83d3f47b3bce769');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/23_outdoor_GT.jpg', 'dd32a30d1029e39c257dc3398759a576', 'WPX/O-HAZE/clean/23_GT.png',
        '0a4b80a1faff728394cc2bf340f2535e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/24_outdoor_GT.jpg', 'd61ececb236ec1a0f4201e053eac3789', 'WPX/O-HAZE/clean/24_GT.png',
        'fa47756d432f9ff166881deb5e3ddfb9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/25_outdoor_GT.jpg', '19eb0858e23c24754cd83ac977430568', 'WPX/O-HAZE/clean/25_GT.png',
        '2f1b494fec435f677a51aaa1531cc8d4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/26_outdoor_GT.jpg', '950ebc2ddf19fa893c479dcd33b42821', 'WPX/O-HAZE/clean/26_GT.png',
        'c976fe01dc0a16655d485f6d25eae6f4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/27_outdoor_GT.jpg', 'a3b4e962178aac4295d07030e11d43e0', 'WPX/O-HAZE/clean/27_GT.png',
        '94662d20a60d41a25ed31076c415bb13');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/28_outdoor_GT.jpg', '86e045b1edae9742348bbabeb823e098', 'WPX/O-HAZE/clean/28_GT.png',
        '21e8f2f7e9a284eefea2658ef8da8e9f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/29_outdoor_GT.jpg', '5f4adb209d5440b323125adc293f795d', 'WPX/O-HAZE/clean/29_GT.png',
        '5b587826123869c5f1c08c00f35d258a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/30_outdoor_GT.jpg', 'eb44813d891c047cff36c21ba2442aae', 'WPX/O-HAZE/clean/30_GT.png',
        'e09bad30a372fa02384847f96b3ee705');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/31_outdoor_GT.jpg', '60e7f620ebe8859ec31fc0e74c2c7a14', 'WPX/O-HAZE/clean/31_GT.png',
        '8c12a63a2ec9a3c8807409de5bea8d85');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/32_outdoor_GT.jpg', '06c6a9692b79f75e8b7824fe0ff25b7a', 'WPX/O-HAZE/clean/32_GT.png',
        'fabc026e31fb329e282b3dc166915af6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/33_outdoor_GT.jpg', '46c6eddb799952441f5eb34916e4d26e', 'WPX/O-HAZE/clean/33_GT.png',
        'ea79d380515c26b0c26489569fd2f078');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/34_outdoor_GT.jpg', 'cbf5c70503871815a61c797c810212e1', 'WPX/O-HAZE/clean/34_GT.png',
        '41767aaa63cbbdb7f95b2d2cb911949b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/35_outdoor_GT.jpg', 'c8e9b7fd38db29307acd13b56880415e', 'WPX/O-HAZE/clean/35_GT.png',
        'e0ec1ff9bcd99e916ee7d4a025f792b6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/36_outdoor_GT.jpg', 'b97e0b97ab1ab767dad191013bc23e02', 'WPX/O-HAZE/clean/36_GT.png',
        'fa55e4f76e52c2742d2bcc4581609f1e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/37_outdoor_GT.jpg', '403cd40272421f8c0958aa3bb5b06936', 'WPX/O-HAZE/clean/37_GT.png',
        '434296a2b6c90cb1d1e0889f05eb93b5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/38_outdoor_GT.jpg', '33cb12fd6f7ac30d3fe21c9795c0338e', 'WPX/O-HAZE/clean/38_GT.png',
        '3d3bcfed2dd1850fea9f24226603ef1b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/39_outdoor_GT.jpg', '1547d147c8b43c88aaffe3d7f0f32ea4', 'WPX/O-HAZE/clean/39_GT.png',
        '8ad10ae8ac90da88a622a4e14ccb8e20');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/40_outdoor_GT.jpg', '604cc086f23930caf6062ac73925ed26', 'WPX/O-HAZE/clean/40_GT.png',
        '9528d5ca7024fc30ea0769b75909583a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/41_outdoor_GT.jpg', '3f48ded28371b1a8301a1007a08dd624', 'WPX/O-HAZE/clean/41_GT.png',
        '8f71c423d2c1874d83ed12106b7934c2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/42_outdoor_GT.jpg', 'd75d84ae8d123f6d6d8d834d718f949c', 'WPX/O-HAZE/clean/42_GT.png',
        'da0e3f095ab336bbabe1a566822ba0c8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/43_outdoor_GT.jpg', '95055a29b746abb2e2aeb1a17730eba3', 'WPX/O-HAZE/clean/43_GT.png',
        'f78980f5202fd8fc31fa8b96e83f91ad');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/44_outdoor_GT.jpg', '5071d932a7111287c3c89aeeb64f2ad9', 'WPX/O-HAZE/clean/44_GT.png',
        '01175f61a8d5a8c5eecc0c5c4db94005');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/clean/45_outdoor_GT.jpg', '3ba77eb08eff6b6f47e40498bfdf8bbd', 'WPX/O-HAZE/clean/45_GT.png',
        '85822b6e1879f1d5ddf7bb6f0f8da251');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/01_outdoor_hazy.jpg', 'a66013c9a0ba9d65b89d58b328092d39', 'WPX/O-HAZE/hazy/01_hazy.png',
        '987190b8c3ee5a8fe31bc4eab9f796cf');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/02_outdoor_hazy.jpg', '37d1f4f6d4dae6c1141f0d5fb674afa9', 'WPX/O-HAZE/hazy/02_hazy.png',
        'c45beee9eebbde52e0d15959a6630a94');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/03_outdoor_hazy.JPG', 'f7a05680ddb48aa55982f887c80512f5', 'WPX/O-HAZE/hazy/03_hazy.png',
        '33bbf4c853f478bcd5b2e2345bac2e95');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/04_outdoor_hazy.jpg', '83949268ab4df85b4a02e7b9f4b84d0b', 'WPX/O-HAZE/hazy/04_hazy.png',
        '5509ebc6bf5df8d30b06846e718b35a3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/05_outdoor_hazy.jpg', '3dd7abd6687aa20d662f423361455220', 'WPX/O-HAZE/hazy/05_hazy.png',
        'c289d052cb82cc7d60e23a28733a9dc7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/06_outdoor_hazy.jpg', 'f25bfa5c329d60775166b07d48c5b8d0', 'WPX/O-HAZE/hazy/06_hazy.png',
        '9712ee4989008aec9d2689e89011ce24');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/07_outdoor_hazy.jpg', '274d7ed57e819e199527978df739904f', 'WPX/O-HAZE/hazy/07_hazy.png',
        '6bf2c22c561765bb803da0c1ad8c1d60');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/08_outdoor_hazy.jpg', 'b307e9a01872e91fba579f603444aba1', 'WPX/O-HAZE/hazy/08_hazy.png',
        'adee2b8cf7404703059e2140602262cb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/09_outdoor_hazy.jpg', 'e418524dd0ce558a4aa9010cbfe1f9a9', 'WPX/O-HAZE/hazy/09_hazy.png',
        '814a2d908254f3b025e6d89676f541c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/10_outdoor_hazy.jpg', 'b340691e1e36477963975bd14144b0a9', 'WPX/O-HAZE/hazy/10_hazy.png',
        'fed3dee25b1665bbc5ba59872440b3f7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/11_outdoor_hazy.jpg', '69c6fb3333bd65f6554ac2197410fdb6', 'WPX/O-HAZE/hazy/11_hazy.png',
        '4c5b72955c4bce57a31aaa484bd84548');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/12_outdoor_hazy.jpg', '17bd548bbdf8f6f5af3998e2494e398d', 'WPX/O-HAZE/hazy/12_hazy.png',
        '63cc0ddaab9eb7514a14639792387451');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/13_outdoor_hazy.jpg', '89c67050ab7d3f838204225c95d8dfb9', 'WPX/O-HAZE/hazy/13_hazy.png',
        '64c19e32ac79fb3390f427b3fbb089df');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/14_outdoor_hazy.jpg', 'af4a6a1f616f28840bf2f284af98fc94', 'WPX/O-HAZE/hazy/14_hazy.png',
        '9115bb90099583953e6faf87347a2b4d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/15_outdoor_hazy.jpg', 'df8d5f4fcf29ec73eb2c762cc793f17b', 'WPX/O-HAZE/hazy/15_hazy.png',
        'edbbe8225b6af345d9bbd46343335c59');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/16_outdoor_hazy.jpg', '1de2862db23a92f2be5a433cb23130ab', 'WPX/O-HAZE/hazy/16_hazy.png',
        'df1328e0a1413a267a21f4931bab0a08');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/17_outdoor_hazy.jpg', '5d84061f55e90e4b82167aed6be81228', 'WPX/O-HAZE/hazy/17_hazy.png',
        '15111cd935ab9fa8314038d6de849b67');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/18_outdoor_hazy.jpg', 'feeacc157a0c5d9e70057821b6852ad3', 'WPX/O-HAZE/hazy/18_hazy.png',
        'b2f5bdea1239c1f26e74896a969157ef');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/19_outdoor_hazy.jpg', '9b7737e6e3512b82013c118644c42db5', 'WPX/O-HAZE/hazy/19_hazy.png',
        '7d6a26037194c3f114e2c1e15b962bd0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/20_outdoor_hazy.jpg', '86dd018e2b6b0adb5786ff036ca6541d', 'WPX/O-HAZE/hazy/20_hazy.png',
        'a530b1661f46fa0b3bc5f0849c5fddcb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/21_outdoor_hazy.JPG', '208f258446e2e63eb4a5f847b8a9ff27', 'WPX/O-HAZE/hazy/21_hazy.png',
        'aea00d03d399a4a3e78c883c7388b66d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/22_outdoor_hazy.jpg', '6200721c7455a3a8200bfe2f900c93f6', 'WPX/O-HAZE/hazy/22_hazy.png',
        '67e97d5c31c0d1d0ab159c0c877255e2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/23_outdoor_hazy.jpg', '3931491859ccb51a776c247b498b4035', 'WPX/O-HAZE/hazy/23_hazy.png',
        '6f975ca3e31729fe302f0ce55e144db2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/24_outdoor_hazy.jpg', '3cf1c12410e2af069b34b84acb444679', 'WPX/O-HAZE/hazy/24_hazy.png',
        '758192ac2012ec8a278d472ebd612df7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/25_outdoor_hazy.jpg', '0cbe34cf10b1f665316833cd361f2d0d', 'WPX/O-HAZE/hazy/25_hazy.png',
        '3ee869a589a64ba7d7e70201b88beee5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/26_outdoor_hazy.jpg', 'a7747b8f521357af2d20a841b3bdcd21', 'WPX/O-HAZE/hazy/26_hazy.png',
        '7ec2adc4bc44709ce14a65686abb1a9d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/27_outdoor_hazy.jpg', '0033e5f1f01accca6d3f53de37713dce', 'WPX/O-HAZE/hazy/27_hazy.png',
        'd02ed956630a34110e5a495c9792787a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/28_outdoor_hazy.jpg', '3fd547affa73ccf9843080e1a5d0006e', 'WPX/O-HAZE/hazy/28_hazy.png',
        '1770808142707a64f855ed51ef3b2db5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/29_outdoor_hazy.jpg', '9647353e4dada89d97f213ee41d6e2a6', 'WPX/O-HAZE/hazy/29_hazy.png',
        '3b5ad8f4f92806d3ed5ba50892444b23');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/30_outdoor_hazy.jpg', 'c564b9ec5be8dbd978106a7b82c19828', 'WPX/O-HAZE/hazy/30_hazy.png',
        '59a5291474d0c20ca23a49b3c12bb93c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/31_outdoor_hazy.jpg', '427b743560d6f8119db4871cda66f5d5', 'WPX/O-HAZE/hazy/31_hazy.png',
        'd860daa7bf584389e0ff478bcf1a9195');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/32_outdoor_hazy.jpg', '0b0f9434df99ad810d43a426c4ef6767', 'WPX/O-HAZE/hazy/32_hazy.png',
        'dcc7ed9a986019ba1217b1aabba1f77c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/33_outdoor_hazy.jpg', 'ce9bca12053070a69b539955f63d1643', 'WPX/O-HAZE/hazy/33_hazy.png',
        '8e0cea6d1f05fc246a87e4a19ed4b11c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/34_outdoor_hazy.jpg', '59ac59e5c2f6590f0c45b8d3f8ebaaa3', 'WPX/O-HAZE/hazy/34_hazy.png',
        '4db0156a0f6402eeddf34365636f531b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/35_outdoor_hazy.jpg', '95c977a8c3a8e6f718ff0e2e24d03106', 'WPX/O-HAZE/hazy/35_hazy.png',
        '0d6cf94fd5ef0f7d4185240b0a5657be');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/36_outdoor_hazy.jpg', 'c891e1d06cc83e8d409b49cce4fa03cf', 'WPX/O-HAZE/hazy/36_hazy.png',
        '38c59c5725218559dd0d01b072e85941');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/37_outdoor_hazy.jpg', '6cdbff8dcb8ee314996d1cf5ef538a43', 'WPX/O-HAZE/hazy/37_hazy.png',
        '9fbd5c4e502a851dd46ee201280c8ff5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/38_outdoor_hazy.jpg', '1de85b17182a681e5c67bc8d88e9cd28', 'WPX/O-HAZE/hazy/38_hazy.png',
        '6f5b873161bc52b323fdfe7d59437797');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/39_outdoor_hazy.jpg', '2f0d2063cc193089cb436fa0ac7919f1', 'WPX/O-HAZE/hazy/39_hazy.png',
        '576c2c6ed06725aec847f6034d0c9db1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/40_outdoor_hazy.jpg', '8b4b2357a6f181164bc92ef03d833e70', 'WPX/O-HAZE/hazy/40_hazy.png',
        '919a14c248a1a9fdc42656e50d116a9f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/41_outdoor_hazy.jpg', '8fc12de3d84ef1e7a356955cd887eb3c', 'WPX/O-HAZE/hazy/41_hazy.png',
        '930e749ced3f82d1944e5f2f59687d00');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/42_outdoor_hazy.jpg', 'a1690afbabaa3a8409eb3747c1f0e54d', 'WPX/O-HAZE/hazy/42_hazy.png',
        '0a9f58ac197f52edd03e4bf94d1669e6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/43_outdoor_hazy.jpg', 'a01edca7b16281f221ee56c9e0da3fbb', 'WPX/O-HAZE/hazy/43_hazy.png',
        'b6ee2d0721bebf63f2a16a854ac869e1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/44_outdoor_hazy.jpg', '549fff889bfe7c104acced9ece6c483e', 'WPX/O-HAZE/hazy/44_hazy.png',
        '4ef90395d237f51929e04db9af66fffa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('O-HAZE/hazy/45_outdoor_hazy.jpg', '11d7b992ef8f57d0055e91c00f9f1340', 'WPX/O-HAZE/hazy/45_hazy.png',
        '50bacaf916a155ac207bf496c8125d95');

# DENSE-HAZE
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/01_GT.png', 'c0dd4261b3e011c3f7dad7cddefa8c78', 'WPX/Dense-Haze/clean/01_GT.png',
        '9d0f7f46137e39650e26defd5f55097b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/02_GT.png', '2294b02f162675c440651ea54e23589b', 'WPX/Dense-Haze/clean/02_GT.png',
        'ff12d1312f8d3984b627ed647088a9a7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/03_GT.png', 'cf8760e6fc38ef9ea5e9ba22a45156f7', 'WPX/Dense-Haze/clean/03_GT.png',
        '85e7cedbd769eaba5f35bf48036cfe90');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/04_GT.png', 'be6fde3e8d8e4594bc86ea1c1fbd79de', 'WPX/Dense-Haze/clean/04_GT.png',
        '37d6d17258dca43c7b57f1f00eeb3102');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/05_GT.png', '73876529185daf8b3b9410fae0876c0a', 'WPX/Dense-Haze/clean/05_GT.png',
        'e2fce8bbf6a86643f042ab71041bf775');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/06_GT.png', '0598646036e4112e36d9ce5bc14def21', 'WPX/Dense-Haze/clean/06_GT.png',
        '6d7b38aa471f8b404cae372d6d13397f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/07_GT.png', 'f0d8cf19fa647f87414d523d2ac596bb', 'WPX/Dense-Haze/clean/07_GT.png',
        '70ef9f9763ddde39c803db646c9a7c23');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/08_GT.png', '422275e2a0b476f0f34a1075f0ff3d25', 'WPX/Dense-Haze/clean/08_GT.png',
        '08ff0722a51840cb8425998af4a6e2f2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/09_GT.png', '64eba4ac37a97ae47182649261d5fe7b', 'WPX/Dense-Haze/clean/09_GT.png',
        '5e0cd923f8b8a3922d6ff5b36b38758f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/10_GT.png', '741eac43ecf63ac7cdf4ddf71079285b', 'WPX/Dense-Haze/clean/10_GT.png',
        '1e2d6c3ea2533451b34195f308f9e9ec');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/11_GT.png', 'd27dd605b516d314a5da5e1d0294e5a4', 'WPX/Dense-Haze/clean/11_GT.png',
        'fc6d48a468fc0079dd987d0cd649086f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/12_GT.png', '1bdb7935f1d8988632653d8b65901494', 'WPX/Dense-Haze/clean/12_GT.png',
        'fb6c91317e7ef978e76625bc5ba4050e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/13_GT.png', 'a52a0eed1fa22fe6c64bba5a730152f7', 'WPX/Dense-Haze/clean/13_GT.png',
        '709931227e30d4007035911ad0c74b7c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/14_GT.png', '140e6dfd6241265362948d0201080a7a', 'WPX/Dense-Haze/clean/14_GT.png',
        'a12c6b659adeca3647de4066d2bfc350');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/15_GT.png', '16abcbe8215d3fdd03d638bb8341d359', 'WPX/Dense-Haze/clean/15_GT.png',
        '80b81744a8ff845a74b0115035b938a5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/16_GT.png', 'e246a6902d3938bcfdfa416bd599f585', 'WPX/Dense-Haze/clean/16_GT.png',
        '156a1a2c50899907ad312e6328957cff');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/17_GT.png', 'ad66fb2d0ce005d2e96e7cb692f44ae3', 'WPX/Dense-Haze/clean/17_GT.png',
        '2800ce66059cd0180c89ae5facc0c2a0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/18_GT.png', 'f7b8c0ec9caf4f6a78de106e7915bd8e', 'WPX/Dense-Haze/clean/18_GT.png',
        '1198c6a02ae58a4231c07685067bbe98');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/19_GT.png', 'f423bc4e05a7a6ef5e48c4d672469d44', 'WPX/Dense-Haze/clean/19_GT.png',
        '878def7055bd494fe80f233cff2cafdb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/20_GT.png', '1b35356127845b157ac841aa8f3938c3', 'WPX/Dense-Haze/clean/20_GT.png',
        '79ca120ca7fec90b2ebdadbe90f1e96c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/21_GT.png', 'b2210e69f7d068efd40bd94136de6e1d', 'WPX/Dense-Haze/clean/21_GT.png',
        'eefac8c131ab8f54daca0bcb89f80bea');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/22_GT.png', 'b1a253ecc8dff338ecbc9110ccf39960', 'WPX/Dense-Haze/clean/22_GT.png',
        'e3acbe43ad921054264a19f8f8fd86a2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/23_GT.png', 'e4d4e08c4bef579fe55a6865ba48a5ae', 'WPX/Dense-Haze/clean/23_GT.png',
        '92416f153dc9beec9356e484c51422f9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/24_GT.png', '0f8d53bc2667781912995229e1bb069e', 'WPX/Dense-Haze/clean/24_GT.png',
        '77dd4b43237da5a2bbe11d020a097869');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/25_GT.png', 'f14d180e035104b7216b8bd110ae01e7', 'WPX/Dense-Haze/clean/25_GT.png',
        '8da418d79c31700cc1a694828e525446');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/26_GT.png', '3e2d2f04fb6b5c35d48ea313b23b3a67', 'WPX/Dense-Haze/clean/26_GT.png',
        'ac0f78e68cff1388cd1dc91876655147');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/27_GT.png', '59ad26eb843c378c29c2a2cbbd977f93', 'WPX/Dense-Haze/clean/27_GT.png',
        'a867d887efa1e20dc4f321673ccc270a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/28_GT.png', '8cfe4dd7b897612850b221d42b270d18', 'WPX/Dense-Haze/clean/28_GT.png',
        '9a460e5780563501b3780a9b3d0b6a51');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/29_GT.png', '679f3cb6d118f98fcfaf7a0f02b00019', 'WPX/Dense-Haze/clean/29_GT.png',
        '64e176b8fa6a7f881c200914ca1bfc6f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/30_GT.png', '697ae77d081c2f782c13e75a9e6a92f4', 'WPX/Dense-Haze/clean/30_GT.png',
        '56b06c458c43c462f6ab807323e9e350');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/31_GT.png', '278722d0325a2d647fc0b28bbee526a0', 'WPX/Dense-Haze/clean/31_GT.png',
        'e61310236f0dee25b0d42ffea1175d84');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/32_GT.png', 'd26fe66c6b09d6201577574b6fd0fac3', 'WPX/Dense-Haze/clean/32_GT.png',
        'a31d7a040868dd6e54dc9a5193cd5279');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/33_GT.png', 'b2c4c4a940d0aad3c6cc20c13c1f7488', 'WPX/Dense-Haze/clean/33_GT.png',
        '3471b53936d6969c6eb408db76d49a8b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/34_GT.png', '2e0453e3a010d0dd36d937dfc0fd9fe6', 'WPX/Dense-Haze/clean/34_GT.png',
        '98ce98aa36017433eda232ae2a1efcc8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/35_GT.png', 'b205390d5189d7b080b27ae76708f087', 'WPX/Dense-Haze/clean/35_GT.png',
        'de4b5eb2f93dc0d4dee287a7a2685d1a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/36_GT.png', '1cbfc29117c35426e70c2b46a753b8e8', 'WPX/Dense-Haze/clean/36_GT.png',
        'd8306dee2ec87d79a6c28f3bf5b3c32f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/37_GT.png', '7b0c252ca3e29eaf72a2e07fdc4949a3', 'WPX/Dense-Haze/clean/37_GT.png',
        '317eee3321d00b96f0170650b2858b55');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/38_GT.png', 'ad22fcd6ea662ee37cda7985db37c56b', 'WPX/Dense-Haze/clean/38_GT.png',
        'fbec17da3747e38d0456e19555237103');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/39_GT.png', '93e801782a56e4cc731d66e60c404dff', 'WPX/Dense-Haze/clean/39_GT.png',
        '629e242d024f168e78e0294da65fe960');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/40_GT.png', '6d3dabab7f4f4e06a060fdcf6e4466fc', 'WPX/Dense-Haze/clean/40_GT.png',
        '19734f16da65bfb6c7821f62bd2eca5d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/41_GT.png', '56e3bd042494a55e282c7ef70d77ea2c', 'WPX/Dense-Haze/clean/41_GT.png',
        '0fcd09bfa4ff20572bc68c944acbb49e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/42_GT.png', '10328ceb345bbbf6a9a1f67b8e822122', 'WPX/Dense-Haze/clean/42_GT.png',
        '45e4650ac858335fabd8ffee2af4ce90');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/43_GT.png', 'a343c4d546398d35cd3b7c1a740db367', 'WPX/Dense-Haze/clean/43_GT.png',
        '74e0b2f1ca82b1dc90b83cf178053524');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/44_GT.png', '730f2c3fa816ccbc923694d8c2485c6c', 'WPX/Dense-Haze/clean/44_GT.png',
        '9dabb039d3e34ec37dc342002e8abd75');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/45_GT.png', 'b8a0cc8bee31c6c8ea5ce1447d06c784', 'WPX/Dense-Haze/clean/45_GT.png',
        '7c028a9b3f1d8ea8c25234c99065128a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/46_GT.png', '1b8d55b3bb2cc314416c78dbbeb510c0', 'WPX/Dense-Haze/clean/46_GT.png',
        '4ea963cd9db2af91f435c3a6a00f45cf');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/47_GT.png', '356b5d489a03119713f913c6a1b3ca45', 'WPX/Dense-Haze/clean/47_GT.png',
        'de80343d4c9e254c16b3008d8b2a4b66');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/48_GT.png', 'd1e17b49b14fffc9fda4d9332dcbf78c', 'WPX/Dense-Haze/clean/48_GT.png',
        'a5f4c446f5c17f718517d1a907c788fc');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/49_GT.png', '046858c9a9830f5c283c768afca49431', 'WPX/Dense-Haze/clean/49_GT.png',
        '42a7a310466fa0b57dcdd8c49a226308');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/50_GT.png', '115004b951735db1ce0953745ffd232a', 'WPX/Dense-Haze/clean/50_GT.png',
        '7a2d8664accbbcbb244af3f5e9eb3e50');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/51_GT.png', '0d96a14d67c16f00486715878eb92859', 'WPX/Dense-Haze/clean/51_GT.png',
        '87002b5ae832cdf220fc5e6b9c039ec2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/52_GT.png', 'e4380f7091392195d0fc7e494e2b8f96', 'WPX/Dense-Haze/clean/52_GT.png',
        'c52678da2fd8a96f3c64ec45fd18183f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/53_GT.png', '5c1ace0d2f9c82a932a2f4b51425e760', 'WPX/Dense-Haze/clean/53_GT.png',
        'c9b3aec8810c03633e96e0415e0947ee');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/54_GT.png', 'f09fd2840ade98155faafece52bf3b66', 'WPX/Dense-Haze/clean/54_GT.png',
        'f7c2bc7f2bd390d57e5a3ed0a55a42e6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/clean/55_GT.png', '58e28ca832046f3018a2d52f6cc7d8ee', 'WPX/Dense-Haze/clean/55_GT.png',
        '0d1fbec651d4f5bcd8fedfe293aa7fe4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/01_hazy.png', 'c6d59ee60d09dc6eb02e20b678ed9fb9', 'WPX/Dense-Haze/hazy/01_hazy.png',
        '6cc0be4f1d9ef48c0e33536bd8b719f8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/02_hazy.png', '5bf5f0e6219406c5e54e20928c15a504', 'WPX/Dense-Haze/hazy/02_hazy.png',
        'eac61865e80cd14281ae05f3cdb878ef');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/03_hazy.png', 'b8b9f487cd307e949f49e947e674d3f2', 'WPX/Dense-Haze/hazy/03_hazy.png',
        '236c537d3e1eb5630c323a1b0d4cb332');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/04_hazy.png', '8783ba8f724a532566da8db12b8160d0', 'WPX/Dense-Haze/hazy/04_hazy.png',
        '5ce0e82e70d2cb6d8971b45b442190df');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/05_hazy.png', '10872937bef382865be9d6ee20d5eb2e', 'WPX/Dense-Haze/hazy/05_hazy.png',
        '4f3cc24340eaa7b5a091f98b4386c181');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/06_hazy.png', '9336c7b581493542aba4383b1bf47ccd', 'WPX/Dense-Haze/hazy/06_hazy.png',
        '0b55963a814ac88faba453a1caf1ace6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/07_hazy.png', '953ba7a4836962b21e1b82c29d4f4a5c', 'WPX/Dense-Haze/hazy/07_hazy.png',
        '66988eedf97045261a389dd5c6d506fa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/08_hazy.png', 'ea3c45fe68b277d56f4ecd41c666e7b3', 'WPX/Dense-Haze/hazy/08_hazy.png',
        'c312a2e0cca10f67964d293242b7fa3f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/09_hazy.png', '9916e48b9bf813a1cfce4812f85899e1', 'WPX/Dense-Haze/hazy/09_hazy.png',
        '12451ff329ecf2e2c4ece3d47d25c27a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/10_hazy.png', '325630cdd47fbdd6ea9869035ab63f81', 'WPX/Dense-Haze/hazy/10_hazy.png',
        '61b024e70a5ca5e38e6cd63ee797a2cd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/11_hazy.png', '83c4fa0c6be74ada36cb7e7ab75b0810', 'WPX/Dense-Haze/hazy/11_hazy.png',
        'c78bb367d08748f5cbc28f6348d0f3e4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/12_hazy.png', '3738ec5df3d8494513078571371517d4', 'WPX/Dense-Haze/hazy/12_hazy.png',
        '4aa79933707b06669963af4786888d4c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/13_hazy.png', 'ea38b4441293c9c84995c23ade32b0ff', 'WPX/Dense-Haze/hazy/13_hazy.png',
        'd3d064f28ab5c72455a35151ce0b1bd8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/14_hazy.png', '61831fad6bc99c44ffa2d2937da4eca5', 'WPX/Dense-Haze/hazy/14_hazy.png',
        '3509e1e2b391d6fa06f598818f9336c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/15_hazy.png', '69064ebbc02c81f384ae57f2a634ceb5', 'WPX/Dense-Haze/hazy/15_hazy.png',
        '7db631b5c34a8b825a8f005b7e2bab0a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/16_hazy.png', 'fb5c935fc7e3dd3587d415ff78ceb3e7', 'WPX/Dense-Haze/hazy/16_hazy.png',
        'f1607832685186d3a976185d679130ac');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/17_hazy.png', 'bd30cfc28176d6c090b73a9fc51619fa', 'WPX/Dense-Haze/hazy/17_hazy.png',
        '27b0e4819758b61d613c4ef1c8a593ac');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/18_hazy.png', '00f1b899fc39533bcd953d95fff34115', 'WPX/Dense-Haze/hazy/18_hazy.png',
        'bce6c2186d3c24d529e8dcf46ddd1f5c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/19_hazy.png', 'eded66549752bce7f2b9a7c8aa424fef', 'WPX/Dense-Haze/hazy/19_hazy.png',
        '2826e0773f27257f146840d30eafee23');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/20_hazy.png', '7a71f7dda837329ffa0718c3f8b0fcf1', 'WPX/Dense-Haze/hazy/20_hazy.png',
        'b554587eb0130b0542bc45315b5e03a8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/21_hazy.png', 'c4cb133cad4a54852c98c80140b8970b', 'WPX/Dense-Haze/hazy/21_hazy.png',
        '2c4fa86f2b3d6078e7fc6fb0a9c2e0ad');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/22_hazy.png', '60d9c5222d990581afe71a2e3b1cf10b', 'WPX/Dense-Haze/hazy/22_hazy.png',
        '390afd982a129821682512f23b3a65f0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/23_hazy.png', '0a67fd8446c5db1c84234804eb2cc594', 'WPX/Dense-Haze/hazy/23_hazy.png',
        '48acb33df318525d0cc814b50ba939c7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/24_hazy.png', '2297fa5bcfc23c121a4b56f18d298cb7', 'WPX/Dense-Haze/hazy/24_hazy.png',
        'bf07cef8a27503a92878eff7568dde4b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/25_hazy.png', 'b53afc43039083b9757351164666a6d2', 'WPX/Dense-Haze/hazy/25_hazy.png',
        '28be812108faa86d7b65d7829b007324');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/26_hazy.png', 'c5eb724c08a48a1c9ca3a20e114e5442', 'WPX/Dense-Haze/hazy/26_hazy.png',
        '4bfd330e215ae1f79c3f4ff8db15a774');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/27_hazy.png', '610ae71f110a1ee15c6a93338c663157', 'WPX/Dense-Haze/hazy/27_hazy.png',
        '229a95c12337ea52b84d1cdd8215f665');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/28_hazy.png', '46471adace7abf7ce9b7ccacbafebb00', 'WPX/Dense-Haze/hazy/28_hazy.png',
        '027745d7416ad2c788ee82df530bf747');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/29_hazy.png', 'a96f7cf82f0d1e62f0db58fcb483d9bb', 'WPX/Dense-Haze/hazy/29_hazy.png',
        '131a8427e610b6cd13fd5a5afa07c335');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/30_hazy.png', 'a7c1d32702739371164fea762d3bb7c4', 'WPX/Dense-Haze/hazy/30_hazy.png',
        'a3a78b3bd3369bcb8127708da4eb2ee1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/31_hazy.png', '3edfc978ef61be7faf9789bebd19344f', 'WPX/Dense-Haze/hazy/31_hazy.png',
        '2ef13d3f00efd9e398588c6351a6e162');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/32_hazy.png', '5d0b882081ee312bcd5667d92d4733b3', 'WPX/Dense-Haze/hazy/32_hazy.png',
        '583b68c3f5a33b49c19aff0c720332fa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/33_hazy.png', '153bcb503dcc446ef56593ec165254cf', 'WPX/Dense-Haze/hazy/33_hazy.png',
        'b7c61ab96281d1b36d5e84f6044cf496');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/34_hazy.png', '050b91dbc2d4e190d45ef1da3578dfa2', 'WPX/Dense-Haze/hazy/34_hazy.png',
        'fca1bee4b776564a361b7e4b45da980c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/35_hazy.png', '4001f4069c82cced87c59c7cbba04fb2', 'WPX/Dense-Haze/hazy/35_hazy.png',
        '63c3e612b7f519821942d445c03602ab');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/36_hazy.png', '43145ffe48c0d30ed0008530f601570d', 'WPX/Dense-Haze/hazy/36_hazy.png',
        '57a1b6ede3761a51389733fd237a2b27');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/37_hazy.png', '6962b82a2fa2cc097d7dcb9d654cdd7d', 'WPX/Dense-Haze/hazy/37_hazy.png',
        'a1afc37987cdf223b757d54124b557a1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/38_hazy.png', '487e12b8ba1d4d0fe350011ca000b1d4', 'WPX/Dense-Haze/hazy/38_hazy.png',
        '5cab77f2717ed075dd79cb944a01ec55');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/39_hazy.png', '8349adeb04f25843ff848675f85f1984', 'WPX/Dense-Haze/hazy/39_hazy.png',
        '4df5ed1bae19d887698ddf6e5931aba0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/40_hazy.png', '996a460806580c86f90971d7a4fd8c84', 'WPX/Dense-Haze/hazy/40_hazy.png',
        '556106f0062cc368e0cd95f0d4c5e520');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/41_hazy.png', 'd62a5076d05d392f1d5164cf6a82263a', 'WPX/Dense-Haze/hazy/41_hazy.png',
        'e9ccd3a0aa39042133c62b25db373e83');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/42_hazy.png', 'fa4f35ec974840948f27cee69f031a67', 'WPX/Dense-Haze/hazy/42_hazy.png',
        '293aa8fb61a1b3e75151b4fd567b7c8e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/43_hazy.png', '49872dbb91326a217645de1157fc5860', 'WPX/Dense-Haze/hazy/43_hazy.png',
        '41d78757e28b28dfc8cabf43e611771d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/44_hazy.png', 'bf3eec55b4825499fae2f6bbd699bc7c', 'WPX/Dense-Haze/hazy/44_hazy.png',
        'acf82072f74f21f8290e678790902f8f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/45_hazy.png', 'dcc0ee7a5fcda749726b8bb1f72f99b9', 'WPX/Dense-Haze/hazy/45_hazy.png',
        '2e6044ed541b3976d69b3378a1336353');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/46_hazy.png', '438568114cd5c206849434f13ba5b435', 'WPX/Dense-Haze/hazy/46_hazy.png',
        'dfd29dc8b3e40f104bb0741f33595476');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/47_hazy.png', 'ed47b07c065f95004c711ed40149aaa0', 'WPX/Dense-Haze/hazy/47_hazy.png',
        '1e9d207b04aaf860bca6a0f4a342fbfc');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/48_hazy.png', 'e590843db7572012c1199ea621cc1897', 'WPX/Dense-Haze/hazy/48_hazy.png',
        'e452a75135015b289e18b4e0c41fa035');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/49_hazy.png', '2d184c0072b6675fa622695c17bc6a50', 'WPX/Dense-Haze/hazy/49_hazy.png',
        '9f4aae8f7505da69359a8f903b419350');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/50_hazy.png', 'c1126da09ae8a00351b44c7e03ee0864', 'WPX/Dense-Haze/hazy/50_hazy.png',
        '73694335329e8b65c5592b18e22455d4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/51_hazy.png', '759175a091d6235fd83302d6779876aa', 'WPX/Dense-Haze/hazy/51_hazy.png',
        'e7139ba62e7361b59f4b32498d1a3898');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/52_hazy.png', 'f12d73fc5d71943ca75fd7a40dd5e996', 'WPX/Dense-Haze/hazy/52_hazy.png',
        'ec00ffd2ecf010bbcdbdb7515e330d89');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/53_hazy.png', 'cde7a671cf9afb30a35fd67c0a953566', 'WPX/Dense-Haze/hazy/53_hazy.png',
        'a4404884cc21a1ffe494f54201561b4a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/54_hazy.png', '6fc37eaaa265fae0613ebef16df30ec2', 'WPX/Dense-Haze/hazy/54_hazy.png',
        '0aaa8eaa0fc92835cc220125c0694785');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('Dense-Haze/hazy/55_hazy.png', 'e7684a9ddc01554443298c0779d8bf43', 'WPX/Dense-Haze/hazy/55_hazy.png',
        'a46fd20c3c68823c2e78dda087ac575e');

# NH-HAZE
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/066.png', 'f7250eff9a7d29080157b743981a6f22', 'WPX/NH-HAZE-2020/clean/01_GT.png',
        '1b59b8f524eb96a3235bba4b7e3b8489');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/067.png', '88aba6336a6ffce1efd67f0c33021ec1', 'WPX/NH-HAZE-2020/clean/02_GT.png',
        '513785e384d3221d1cb0967ef797039e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/068.png', '0288707d6a76d1f5e359c01248ad6a36', 'WPX/NH-HAZE-2020/clean/03_GT.png',
        'bd63bd286a6d53c6834feddaa3897425');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/069.png', 'da0d846e8cf1f30c776042c66548903f', 'WPX/NH-HAZE-2020/clean/04_GT.png',
        '24e3fa120a370f770c3bfa14dd74e5f2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/070.png', '057cc78c6d4dbfb440992a146509c561', 'WPX/NH-HAZE-2020/clean/05_GT.png',
        'b74d7db0b590062b59c039ea211169de');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/071.png', 'acc099ddb8345dbbc772fab086548b13', 'WPX/NH-HAZE-2020/clean/06_GT.png',
        'fb55be5c6880c6621d8fb77f5b5d43a8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/072.png', 'd10da8110588e703f0f4c67206acde4a', 'WPX/NH-HAZE-2020/clean/07_GT.png',
        'f94e93ffef1b02f994449a40a83e1240');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/073.png', '1d0e200ed3f0423b33c4555ca3017f9c', 'WPX/NH-HAZE-2020/clean/08_GT.png',
        '706ee298c2532d59997a82eb3b3a6206');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/074.png', '5962ba1d501a3f770811a878de836e64', 'WPX/NH-HAZE-2020/clean/09_GT.png',
        '7fe5faaa99acd4e959b7f1f5e6c2cd94');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/075.png', '3500da8175de4b15af118468173ce00e', 'WPX/NH-HAZE-2020/clean/10_GT.png',
        '72ca0c1c99103d7ad590643f83d97185');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/076.png', '5c861157c5dea0c8bbec4fbd018ed92c', 'WPX/NH-HAZE-2020/clean/11_GT.png',
        '7d57fae3396dbfbc858c442b4234bd59');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/077.png', '228507a40c9604bdefff44a43d1b1feb', 'WPX/NH-HAZE-2020/clean/12_GT.png',
        '475b2d06f6372a5634c5ca5d0b92dc81');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/078.png', '0525aa59a2f252d091116f2127ca519e', 'WPX/NH-HAZE-2020/clean/13_GT.png',
        '9296e2b30ae5565ac84c0c354854d2fa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/079.png', 'e827252ed1d0477ccbba3bbae17b3635', 'WPX/NH-HAZE-2020/clean/14_GT.png',
        'e8410fcf0ce694571e9a6f88e7772ec2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/080.png', '9fa9d8c3cdc41c0d98447194fea63d99', 'WPX/NH-HAZE-2020/clean/15_GT.png',
        '0e757695496a7bf3982b74ccfc4ac211');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/081.png', '00722d447ee20c1032f1bf60767be197', 'WPX/NH-HAZE-2020/clean/16_GT.png',
        '2ecc47f9d780e9c722533d92a6ad897c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/082.png', '56177b06123ebf07af8e29330fa9fda8', 'WPX/NH-HAZE-2020/clean/17_GT.png',
        '10a0ba91078d22835803fe41995cbe08');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/083.png', '34312faed14b508b1b8d1f1b6a07afc9', 'WPX/NH-HAZE-2020/clean/18_GT.png',
        '21a98bf39baf4f10c2a767cc069772db');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/084.png', 'db701eb32041c200c728c8803aefaedb', 'WPX/NH-HAZE-2020/clean/19_GT.png',
        '023cc1773901e40559eeb613a9f9fde5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/085.png', 'f04066b122d3a0309665b19ee4d1b3bb', 'WPX/NH-HAZE-2020/clean/20_GT.png',
        'a23e13c1bb8cf11f3a14139fecaf274d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/086.png', '801a997d69578844643da1f0be96ffa7', 'WPX/NH-HAZE-2020/clean/21_GT.png',
        'f72c0d6bd40b2993952925a99fe90db5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/087.png', '11c09f4d7c98cca6ecbe96f6803d5fb1', 'WPX/NH-HAZE-2020/clean/22_GT.png',
        '01751ec25c6aac5aca3fd457783fddf0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/088.png', '1a4a1621938a1050ea9a71db3a723f67', 'WPX/NH-HAZE-2020/clean/23_GT.png',
        '4d3ef13581c41c577e96a53c12c0cf2d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/089.png', '149f8361f5b16d47f3cc2be25e09ec2c', 'WPX/NH-HAZE-2020/clean/24_GT.png',
        'aa7a90cf7b6dbf6b8c97ecc1b4aea40e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/090.png', '0cfbf593e8cd6f9cda245ecea7dda902', 'WPX/NH-HAZE-2020/clean/25_GT.png',
        '5ba55737e6ebfdb2c918f433d03a6849');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/091.png', 'e73b886cd3c93b131b8ae3d6bcaab684', 'WPX/NH-HAZE-2020/clean/26_GT.png',
        'd846b389a15e5d73fb8dd71607cb6cf3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/092.png', 'a07292fa7d398ad417c73ee84c482994', 'WPX/NH-HAZE-2020/clean/27_GT.png',
        '6c71bb9a9033f7209b565a45ace5e45a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/093.png', '813005064246de209af0e6c52eb48e72', 'WPX/NH-HAZE-2020/clean/28_GT.png',
        '53f6822ba719b67ba62748ebc577a1a8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/094.png', '72421ebde10893cd1beaa72262c53555', 'WPX/NH-HAZE-2020/clean/29_GT.png',
        '1087378c9407ccbf3a57925e5d784d72');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/095.png', '87b63d98c2ebc7210f81d24765a7c560', 'WPX/NH-HAZE-2020/clean/30_GT.png',
        '08323e27b1457e44c7b734c69a82f082');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/096.png', '4b7b9be7c355c809d5c7233fe9bcbc44', 'WPX/NH-HAZE-2020/clean/31_GT.png',
        'e06e43d3c128eaa5200f3da012421c5a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/097.png', 'e890f27bf03dbbba0a68873372746d5b', 'WPX/NH-HAZE-2020/clean/32_GT.png',
        '0479dcfa39528ffb6bd7705e0d21a1c3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/098.png', 'f989b4ca431c6a589935b47b76dfd1b1', 'WPX/NH-HAZE-2020/clean/33_GT.png',
        '15475000ff6eaacbd6545bc8ca44d0cf');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/099.png', '83c1b781cecce612fd285b1526b05ac4', 'WPX/NH-HAZE-2020/clean/34_GT.png',
        '6d991de8fd6ae4e69c9483b7e0cc87f8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/100.png', '1f5b0d43f32f744828e39d1de9fddeed', 'WPX/NH-HAZE-2020/clean/35_GT.png',
        'd6a06ff7dfc55ad75a852a3e201e83df');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/101.png', '106f2e04b522e894822214fe3b24150d', 'WPX/NH-HAZE-2020/clean/36_GT.png',
        'ce9b64fab29f82be88f283564500c7a5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/102.png', '8f47f8bb8ee8483c91e251a2902be6cd', 'WPX/NH-HAZE-2020/clean/37_GT.png',
        '3f8bbde797dd3cd5db6d0d3617a947cb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/103.png', 'b82ee1fb759166d3c8bf4881d9b9902e', 'WPX/NH-HAZE-2020/clean/38_GT.png',
        '5ff7fcbdbbafebff328e964156de634b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/104.png', 'd33bfff95627b57413e18cf6c4aeb59b', 'WPX/NH-HAZE-2020/clean/39_GT.png',
        '6f88ff00ba18b859f3e4006f61f6f934');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/105.png', '891a754972828d4326e42a89a49b1a2c', 'WPX/NH-HAZE-2020/clean/40_GT.png',
        '4ccc718e703959499b93d62ab1e17638');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/106.png', 'b5f93e30c856381c844a1228dbb3ea4a', 'WPX/NH-HAZE-2020/clean/41_GT.png',
        'ae9b330c1da6fda44f7884ea6bd6d747');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/107.png', '0e049280e1991554756ab4e7ebffd4ac', 'WPX/NH-HAZE-2020/clean/42_GT.png',
        '8ac3ecdfbb056b4f31827af884aff5d2');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/108.png', '2643ab70dfae7126d25797b5075d9f3f', 'WPX/NH-HAZE-2020/clean/43_GT.png',
        'b2cd6a2a57b8cbb228b71dfacaff7ee8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/109.png', '95fd16b2985ae8bb54b376e13b3c9df5', 'WPX/NH-HAZE-2020/clean/44_GT.png',
        '84fa358e7561d708b9d923f404000e5a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/110.png', '00562f1ad9ffc09003bcb0eca29ab68c', 'WPX/NH-HAZE-2020/clean/45_GT.png',
        '663d87e8bb5b0e8a6b1737aabaeefb73');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/111.png', '9a92497cb28ca1fa27760731df29ae32', 'WPX/NH-HAZE-2020/clean/46_GT.png',
        '24e077293669fa3360ac58f08e667f9f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/112.png', '5188f966562076e23cbaccc57c8eeebd', 'WPX/NH-HAZE-2020/clean/47_GT.png',
        '9b7fd5e2d781fa89bccf7ca8031106a0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/113.png', 'fec9de0b49c4fa0c9ca1cb848b465591', 'WPX/NH-HAZE-2020/clean/48_GT.png',
        '20c5c5e2ff2d7fc00091f981ab0025c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/114.png', 'a25bb936ad61c007b86334ff8d746abd', 'WPX/NH-HAZE-2020/clean/49_GT.png',
        'd1cdc38b0c540258a938a9ed54af29a4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/115.png', '9cb8217b18314dd318632258e3817aa1', 'WPX/NH-HAZE-2020/clean/50_GT.png',
        'b61f0373db3a18970b4fd9814c879b72');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/116.png', '6e1bb4ece16c4989823ca7dc15703ea4', 'WPX/NH-HAZE-2020/clean/51_GT.png',
        '9e17ccad7ef0d0f5af5a216a18cc8bfa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/117.png', '0db24f8a3b6036d31781c35b350f0006', 'WPX/NH-HAZE-2020/clean/52_GT.png',
        '65fafa16096217cf2c56d8ac1c7c7d08');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/118.png', 'd050c1ae62f99923b18aa75b9a6d456e', 'WPX/NH-HAZE-2020/clean/53_GT.png',
        '403e2ef07558b2423e933443530cc67c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/119.png', '5c75d83d1b45d0cab4a56a3cc6963f00', 'WPX/NH-HAZE-2020/clean/54_GT.png',
        'f0fb3d6c60f037b2c82b0c41edae101f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/clean/120.png', '55dd1b2e0027c86b4e7f9ec9d245e298', 'WPX/NH-HAZE-2020/clean/55_GT.png',
        '13accf82fb582fba649453b25838670f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/066.png', '6db6877b0ee81287fed86fe1dcaf44c4', 'WPX/NH-HAZE-2020/hazy/01_hazy.png',
        'd0d13efa1cc491e4164b55edf968dde5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/067.png', '67522608c02f83716aef88ec1138d678', 'WPX/NH-HAZE-2020/hazy/02_hazy.png',
        '8ae4a168916d100b5fd3334b31affe99');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/068.png', 'f1f669a8ccc5690535675b39527a6814', 'WPX/NH-HAZE-2020/hazy/03_hazy.png',
        '4fce83e39578dcd59b5c5dda1d472395');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/069.png', '7b8046e0b3165d413e6c89db903c3f2f', 'WPX/NH-HAZE-2020/hazy/04_hazy.png',
        'c21f50006d0749973ee04f290c40714b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/070.png', 'cfc81113f40dea12beb57ae581ccfa46', 'WPX/NH-HAZE-2020/hazy/05_hazy.png',
        '95d591171bd4efd4a4f5cdbdead72255');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/071.png', 'b1f1482cacaa048edcf29c383b8ddefa', 'WPX/NH-HAZE-2020/hazy/06_hazy.png',
        '247b2d2587426317529ae634bcd9d3d5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/072.png', 'd7a3b2ee24ac79bd05861cad8af6809c', 'WPX/NH-HAZE-2020/hazy/07_hazy.png',
        'fdf2b270797bbde58554bc229c7dfd81');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/073.png', '756156b91a8068bbec0405ec3de39c02', 'WPX/NH-HAZE-2020/hazy/08_hazy.png',
        '89de63acef8d3f33939197d1911d2c9a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/074.png', '83b263d69fc6b1c28b98173130a3b725', 'WPX/NH-HAZE-2020/hazy/09_hazy.png',
        '7b5f32e3d263c92257f529ec45385df7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/075.png', '2de64fd762111751f0a52246ee23f226', 'WPX/NH-HAZE-2020/hazy/10_hazy.png',
        '2e8e5352a2285b08dc79cfa3f64db87a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/076.png', 'd0fbff0d9f81ff145e1653d505a9fe56', 'WPX/NH-HAZE-2020/hazy/11_hazy.png',
        'cb4e74c86b3b715f0024146e41ca9558');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/077.png', '035ad5136b79cce61433956ef9344875', 'WPX/NH-HAZE-2020/hazy/12_hazy.png',
        '916878f96eddd2919ea892b19c577d83');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/078.png', 'b8015dcf6106188b62ad1eba4346f9e2', 'WPX/NH-HAZE-2020/hazy/13_hazy.png',
        '4088977de1937c9547aab65b27d0e1db');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/079.png', '3f8124206ba8a49249e4814167af3bd5', 'WPX/NH-HAZE-2020/hazy/14_hazy.png',
        'f26af00297b32a3718e535c32e5d3747');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/080.png', '3acb221950bb682c3ac3f12e5c3daa99', 'WPX/NH-HAZE-2020/hazy/15_hazy.png',
        '70224e687deed2799bef4d8d68bbf20b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/081.png', 'e863d767eb130bd81b6ffea10fe63ca7', 'WPX/NH-HAZE-2020/hazy/16_hazy.png',
        '001c0fc635380cb3a8c9ae4c4416deb4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/082.png', '6f4dbefb9af54597791e2ef5169f3f38', 'WPX/NH-HAZE-2020/hazy/17_hazy.png',
        '2860f53a8f7703a1c804e1f0d3db39b6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/083.png', '17583184edb33482ad39eebe335b632f', 'WPX/NH-HAZE-2020/hazy/18_hazy.png',
        'cd69d2ddfed929641ca605c63a7886fd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/084.png', '8aa2284a62f32418e0953e1c30539b38', 'WPX/NH-HAZE-2020/hazy/19_hazy.png',
        'cc7b75675cb554af2f45b5af2ce39277');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/085.png', '0b0b5943a495623d5396f3812b1c5c9a', 'WPX/NH-HAZE-2020/hazy/20_hazy.png',
        '9de8bffb776a9fb00c3b108c423a4ef5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/086.png', 'fc27e040ecd46c8fa416be460224e27c', 'WPX/NH-HAZE-2020/hazy/21_hazy.png',
        'a132de643ee214e839b0a900994c5e76');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/087.png', 'd31a9608e68f6a8f6c0a8bc4d0a49fca', 'WPX/NH-HAZE-2020/hazy/22_hazy.png',
        '89b11bcaab71d16183a0d8843387532b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/088.png', '80e6d6bf1d6ba91d47bbd4d2c2ab92c4', 'WPX/NH-HAZE-2020/hazy/23_hazy.png',
        '19848b25a7c86c85ca56bb7d90b489d6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/089.png', '2aae436e347d798f3d1a8f26ca112012', 'WPX/NH-HAZE-2020/hazy/24_hazy.png',
        '806ba77392baf4358f4087a3fc4de89c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/090.png', '84d984bb90e47fa6233105edf1ded609', 'WPX/NH-HAZE-2020/hazy/25_hazy.png',
        'd77860ad0521e27ac7f14db3a4e89ff3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/091.png', '611d303971854022ad7392d008d72626', 'WPX/NH-HAZE-2020/hazy/26_hazy.png',
        '580a2baf92e509a5b0eefb6c9c9635a6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/092.png', '1c366b120b8fa6c1cd59e8e8590e1e46', 'WPX/NH-HAZE-2020/hazy/27_hazy.png',
        '946a8d303ac60ca8f017b3e78316e239');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/093.png', '8d9031ed73a5885378d50aeb9e02605d', 'WPX/NH-HAZE-2020/hazy/28_hazy.png',
        '5fbcb3f43b0a1c4d11d2d7c3cfeff411');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/094.png', '26ef0a5c1e50966532060c88e924db13', 'WPX/NH-HAZE-2020/hazy/29_hazy.png',
        '50b8a09e3c58d3d85600b8fce01e292a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/095.png', 'fd098512995fac720788c265679ff935', 'WPX/NH-HAZE-2020/hazy/30_hazy.png',
        'bccebc361da2e17232cbf364fe2c03f4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/096.png', 'e131b897cf68d869fdbcb03810926e17', 'WPX/NH-HAZE-2020/hazy/31_hazy.png',
        'eddb22d8bc1fef3f4a448b41c3b1c3d7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/097.png', '73d99c88b804a750b33842d511d2b2da', 'WPX/NH-HAZE-2020/hazy/32_hazy.png',
        'bdadf41bccb85f1551239934a8c07edc');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/098.png', '7eb6f6a577bcb221c3a2ef4481c5a8da', 'WPX/NH-HAZE-2020/hazy/33_hazy.png',
        'd3eb9bd6b0da4f9f94e707f48e954e4e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/099.png', '49826adacb54d9cce9102e6b4b44b5a9', 'WPX/NH-HAZE-2020/hazy/34_hazy.png',
        'ce22f76594db4558ee76ac5469a77d28');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/100.png', '9704f2c321731e931fbb1e3196a56ad9', 'WPX/NH-HAZE-2020/hazy/35_hazy.png',
        'e1dbfb0caa35a031759b390df67694c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/101.png', 'e63231fa72af37be7fec82533a16c54f', 'WPX/NH-HAZE-2020/hazy/36_hazy.png',
        'bf0df9d8b0ab163bc39ff0c087210f9d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/102.png', '9604382f2f10fed3e4181be66e56d54b', 'WPX/NH-HAZE-2020/hazy/37_hazy.png',
        '4f494904910ecea0b727f83155f7cb67');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/103.png', 'ec94ed77fe50bfb11291a7ab9697df20', 'WPX/NH-HAZE-2020/hazy/38_hazy.png',
        '4bc00dbc5ee0afbf4c30e4f57d174fbd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/104.png', '9a2a5065bbdbfe78c5cfd365309f14a7', 'WPX/NH-HAZE-2020/hazy/39_hazy.png',
        'c79d368d647868a7e39bcc3c2f3d36bc');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/105.png', '245fc95d37fe75389d2316ef5f92be95', 'WPX/NH-HAZE-2020/hazy/40_hazy.png',
        'da2450a44cacd5b8b7b5be61d373b481');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/106.png', '90c98f2cf05a689f0c09021fd9e4a981', 'WPX/NH-HAZE-2020/hazy/41_hazy.png',
        'c8bbce269c040f425929738d1d5b9507');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/107.png', 'b6a3dfebd6f48318a51a558b1b5da2b7', 'WPX/NH-HAZE-2020/hazy/42_hazy.png',
        '7da44c0d069cb8014133ba44a8257b37');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/108.png', 'f6cf9b6c642d0abc394a3504764175f6', 'WPX/NH-HAZE-2020/hazy/43_hazy.png',
        '12a0a1726f4cd6dbb3f449863db0906e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/109.png', '44ae99324069b06d260d4e4b69a509fc', 'WPX/NH-HAZE-2020/hazy/44_hazy.png',
        'f79e2bb35f41a9f6f0b7b70e64295699');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/110.png', 'f2869b1ccead1c0049c482558cc461a5', 'WPX/NH-HAZE-2020/hazy/45_hazy.png',
        '078875b6cad44dfed3bc4b000cffdb25');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/111.png', 'f53286b01059a76334cd8d87f973a202', 'WPX/NH-HAZE-2020/hazy/46_hazy.png',
        'f7e0674bd8d1a640b0158aa644138eb1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/112.png', '3e60367aead21d8c00128385e03c4be5', 'WPX/NH-HAZE-2020/hazy/47_hazy.png',
        '93443cfc39651ecf7594ffc84245d1e9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/113.png', '7ae819330b926c302cf7762893ffdeaa', 'WPX/NH-HAZE-2020/hazy/48_hazy.png',
        '35fcd65d9330d5bbbe606313b14b4290');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/114.png', '5f30a4c5717988da1ae9849453871c41', 'WPX/NH-HAZE-2020/hazy/49_hazy.png',
        '624ee50b3076280290f6208719214fbb');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/115.png', '824004e9f47bf07501f4bef864c8d10e', 'WPX/NH-HAZE-2020/hazy/50_hazy.png',
        'a40527358438f9af1561a0508ce1723e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/116.png', 'c11238db385be535b4cb9fcdc4be8bd1', 'WPX/NH-HAZE-2020/hazy/51_hazy.png',
        '181cfe91306a0e9743d1c92f71a0599e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/117.png', '80eae10462bddbbdbfc02ac2c7d57d28', 'WPX/NH-HAZE-2020/hazy/52_hazy.png',
        'f9b11bc0576148fcbc7912edf5a1c0e9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/118.png', 'e7f4cb32585d2f85b4743a62b430faca', 'WPX/NH-HAZE-2020/hazy/53_hazy.png',
        '3f7828205ad6952dbd758ed01fc7161a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/119.png', 'ae8864c33fa57cfbf9bea533d2f33f58', 'WPX/NH-HAZE-2020/hazy/54_hazy.png',
        '3525fe75b751317325c53c6d6f79f9cd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2020/hazy/120.png', '658c85979ea17f3f7cbb819f5ce74a48', 'WPX/NH-HAZE-2020/hazy/55_hazy.png',
        '6acf43aec6aeba73a5d7d2c5d7c8510d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/041.png', '86841f60a043c1f675b3a9f3bb01da20', 'WPX/NH-HAZE-2021/clean/01_GT.png',
        '295ce180f768ba5352d4616c08098e8c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/042.png', '0931061fc2325bce6ea8b5d553b5ee40', 'WPX/NH-HAZE-2021/clean/02_GT.png',
        '80e5b46cee22961d3c51d685f585f5e6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/043.png', '32f66fefcacb61044d7d535a21934654', 'WPX/NH-HAZE-2021/clean/03_GT.png',
        '6bcc4720d9cb5209b6615074e69483ad');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/044.png', '5be104a30b6c748ee6d047237e0e7a97', 'WPX/NH-HAZE-2021/clean/04_GT.png',
        '9f3355a7fc2d39607e5dc0e89d81bfab');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/045.png', 'e41c3df5e9ba48120f4b23d03ff299ec', 'WPX/NH-HAZE-2021/clean/05_GT.png',
        'd3e365eb2d1c799a36ef10e5c75fbc44');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/046.png', 'cedc6b65c0fb5588af1d164969038d35', 'WPX/NH-HAZE-2021/clean/06_GT.png',
        '1a7e7c90c948b3c0a72985649a824f81');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/047.png', 'f0d76a2ebb26e26cd9e136c6c39f23ce', 'WPX/NH-HAZE-2021/clean/07_GT.png',
        '3041c4bdae60c0002174c502b48d4451');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/048.png', '86f1e0a8bce0135e63a45be3f4b45d74', 'WPX/NH-HAZE-2021/clean/08_GT.png',
        '10c1d00e834fc710a2dfe33d74935e53');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/049.png', 'f6d5cfbb368fb1ca2a8bcc702c324fef', 'WPX/NH-HAZE-2021/clean/09_GT.png',
        '15fd6f0329e23624bb1980865745311e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/050.png', '97b86dd3659be8b21a1e465bc9507f09', 'WPX/NH-HAZE-2021/clean/10_GT.png',
        '57a1fd8c5d6e998ac5ef0adf903e3395');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/051.png', '12f21706044c43c8ffce27e3a4f7498e', 'WPX/NH-HAZE-2021/clean/11_GT.png',
        '8d9765e262b7a2cc1ddf0c7c68d0484d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/052.png', 'f68493cd9e159966febb69a9a651d40c', 'WPX/NH-HAZE-2021/clean/12_GT.png',
        'ab916ec6d90b3e62c64c0c5a0272aea4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/053.png', '9cd8dd45fb392946bd9799e86a97abe7', 'WPX/NH-HAZE-2021/clean/13_GT.png',
        'ed5cc86d681bcbb9d52fad008f2ddda8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/054.png', 'f83cb0b40cb1b299a687e03bd80821f3', 'WPX/NH-HAZE-2021/clean/14_GT.png',
        '44b84bb457059000af4ef166090ab2c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/055.png', 'b0afbae0152afea9ca38a7b6e5200b8e', 'WPX/NH-HAZE-2021/clean/15_GT.png',
        'adaefb14e038053541b43788968635c0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/056.png', '21931975bd99854e78f58fc6a209f429', 'WPX/NH-HAZE-2021/clean/16_GT.png',
        '84663e55f2ae191c90a2ea09b41a08ad');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/057.png', 'deb6ad09292140448f3b184beb825c2f', 'WPX/NH-HAZE-2021/clean/17_GT.png',
        'fbab15ea41e561fde21f5690fbd6a33c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/058.png', 'a4347f01ce743dd7c4336b55905afa32', 'WPX/NH-HAZE-2021/clean/18_GT.png',
        'eeb9325c093b4c3aa17c646b54180a3d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/059.png', 'a1eefb51a5692c8a0bd5c0843c081ac7', 'WPX/NH-HAZE-2021/clean/19_GT.png',
        '437ce3d6a88affaa198184d479961d73');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/060.png', '0b3db4c7e32f7c0f78bc1266616dca78', 'WPX/NH-HAZE-2021/clean/20_GT.png',
        '81c78a71879898426887663fe8ce864d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/061.png', '6b2b1e49931abc96b348810688a2b7b7', 'WPX/NH-HAZE-2021/clean/21_GT.png',
        '747f6c518215fb0964a0b54b3c875608');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/062.png', 'b155b17d6486ab503fba1846d81c1732', 'WPX/NH-HAZE-2021/clean/22_GT.png',
        '311d0cabd6522d91911e3f54fd2d5ef5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/063.png', 'ba50bb56fcc12423068865f74652f7cd', 'WPX/NH-HAZE-2021/clean/23_GT.png',
        'bf5ca2dfe9d930f620d486e5b4d7fc77');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/064.png', '972e0f140d687db21ca03ba7ee050a6e', 'WPX/NH-HAZE-2021/clean/24_GT.png',
        '6e2a8a56c098c319e4093fd437903ad6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/clean/065.png', '3287b7a9eb42cf8b388571254f18cfa4', 'WPX/NH-HAZE-2021/clean/25_GT.png',
        '33d9808c45e1bb9bae173a4014b9b4db');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/041.png', '5a703161b89865978e07e99651d01400', 'WPX/NH-HAZE-2021/hazy/01_hazy.png',
        'd15d916f4506598621cd8a4b9da0daf5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/042.png', '78555d0d36d6c4d385bc2ebc37025a04', 'WPX/NH-HAZE-2021/hazy/02_hazy.png',
        '63ee3d72c0573984f33116c9cb03f227');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/043.png', '3009c63faa63eb9a6e48a4d338949192', 'WPX/NH-HAZE-2021/hazy/03_hazy.png',
        '4a8ebee458da3c6a0506e6dc1b59c365');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/044.png', 'c40e7c711e6fe1f3850d87ad9acc5225', 'WPX/NH-HAZE-2021/hazy/04_hazy.png',
        '6cba80d38238ce9c3fca0dd7864f9f0d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/045.png', 'a1868ee1c188ffbd90b639cba4438161', 'WPX/NH-HAZE-2021/hazy/05_hazy.png',
        '89b77e3f8e3fdb7db8de159d6c3da20b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/046.png', 'f2f59f7b1f4a907fa3a3d5f20e68aa41', 'WPX/NH-HAZE-2021/hazy/06_hazy.png',
        '49e859f0c9166e616eeb64094120fcd7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/047.png', '9093a1bf24a3f8653ab7002b57456ff2', 'WPX/NH-HAZE-2021/hazy/07_hazy.png',
        '5316f4770c32be27ac838cce18aa37ff');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/048.png', '29994ebe377d2248ea3c759b1e16e606', 'WPX/NH-HAZE-2021/hazy/08_hazy.png',
        'edcc2b279a746bb658112223496bc21c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/049.png', 'fc3af5266ccf540fba2f8e227d7f8983', 'WPX/NH-HAZE-2021/hazy/09_hazy.png',
        'd515c91bae94695d75ea8fcffdced1c9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/050.png', 'ad367fdb8cb194d5e6f589e7f80002f6', 'WPX/NH-HAZE-2021/hazy/10_hazy.png',
        'd4aeb1f5d57d5a22c3f5198cbfa745ab');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/051.png', '7719c5636f32a31d528b5964741fd44a', 'WPX/NH-HAZE-2021/hazy/11_hazy.png',
        '690a30bb7f0f9771cedd91dba9dbb5a0');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/052.png', '1e3ba53106f4dbbb2c5f16d923d6bbed', 'WPX/NH-HAZE-2021/hazy/12_hazy.png',
        '5f22f3839c1118d7a1ada5c36aa6ae61');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/053.png', '3686150bdcb1a375ee88f5e2fbe1a39a', 'WPX/NH-HAZE-2021/hazy/13_hazy.png',
        '59b3520a1c328997356132a970f46142');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/054.png', 'e1f7aeac7c189f51ce4ec3533fa36b79', 'WPX/NH-HAZE-2021/hazy/14_hazy.png',
        '59e5e65aa122884728ddb39a0d7449de');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/055.png', 'e98cf0a4bf60056424e9c83bf3411f9a', 'WPX/NH-HAZE-2021/hazy/15_hazy.png',
        'c8e40ae0be71407fe1d26e1fbed783ef');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/056.png', '374675e542e6866241e44d66282f84e1', 'WPX/NH-HAZE-2021/hazy/16_hazy.png',
        'ef67fcd621462952a3538c9b2253e59d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/057.png', '3bb7b52f9281bf0aa3cc56ff8ad83b11', 'WPX/NH-HAZE-2021/hazy/17_hazy.png',
        '932c8745bee95bd48c664c55d4e8906d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/058.png', '43a17292267ccdfce1180d027613ffc1', 'WPX/NH-HAZE-2021/hazy/18_hazy.png',
        'b6b3ee248936d8a932b6a81eb93f5494');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/059.png', 'a48d3781c392b0fff4c8aabdfe437eac', 'WPX/NH-HAZE-2021/hazy/19_hazy.png',
        '8b7b6e9c39e960d24e2896a8a627fe8c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/060.png', 'd9b13c370fc4ac4d5d0fdc9f4128c8e7', 'WPX/NH-HAZE-2021/hazy/20_hazy.png',
        '273f5773f4cc1a09f27c686e9de8ab43');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/061.png', 'dc5f8c619fc35897ad921eaa821042e6', 'WPX/NH-HAZE-2021/hazy/21_hazy.png',
        'f1d443368331285ec2543310643756e8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/062.png', '2c0abbe04f316cd6bb8d7d0b644e0afc', 'WPX/NH-HAZE-2021/hazy/22_hazy.png',
        '17571ec45302ef7c9a2d0232e7be3a4a');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/063.png', 'f9e9462cecaf488d14cf57a6fb8cff29', 'WPX/NH-HAZE-2021/hazy/23_hazy.png',
        '1edefc880fbc660ba5a0bc7e1cea7df3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/064.png', '2cc1d4c8379beacfd775cac8ec46ef1f', 'WPX/NH-HAZE-2021/hazy/24_hazy.png',
        '67593a45e8e0d54950c878e03b1643f5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2021/hazy/065.png', '391b9cae1b596e79cd48851d764b0ac3', 'WPX/NH-HAZE-2021/hazy/25_hazy.png',
        '7badf552bc4be76a969ecd245e922303');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/001.JPG', 'f810fe84018d2d1db3044cde7ebdcf66', 'WPX/NH-HAZE-2023/clean/01_GT.png',
        '2a6ea3939eb4a2140b0c2e3b1f9cf514');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/002.JPG', '26193f171f96005150816c96e78bbe07', 'WPX/NH-HAZE-2023/clean/02_GT.png',
        '325917682dff1aa17171ed3a5f7d5887');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/003.JPG', '23fdb521b9731227519810f0773c2e97', 'WPX/NH-HAZE-2023/clean/03_GT.png',
        'cfddd070da00f4232dbd22c3424c89dd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/004.JPG', '80be0c9f15ff89f41d86fd7ab4735d63', 'WPX/NH-HAZE-2023/clean/04_GT.png',
        'fa523aac223a75faf788f711f9222118');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/005.JPG', '88face68a5b40a91a81160c7b0d8a5c5', 'WPX/NH-HAZE-2023/clean/05_GT.png',
        '40aa19d3a7eb5b607168ed4e39509869');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/006.JPG', '3935f7503bbbdbbef375b0e4f741ce6d', 'WPX/NH-HAZE-2023/clean/06_GT.png',
        '55edd9739c822d8eb9ede7e58913de03');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/007.JPG', '9c35066253682c4a8d561f7248dd548d', 'WPX/NH-HAZE-2023/clean/07_GT.png',
        '5c0965aaeb2f94acfb0d134ec389e772');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/008.JPG', 'cb4f15671bf7dbaa112ea6944bf94b55', 'WPX/NH-HAZE-2023/clean/08_GT.png',
        'dc049b8fd14d310bedd3e0932290202e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/009.JPG', '0c09ee937f4dd0632896a33cdef58b04', 'WPX/NH-HAZE-2023/clean/09_GT.png',
        'd59b1cecdc9e23040bf0b4cbb1aa082d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/010.JPG', 'f399e68cd46f4728a88d37a7619e8edd', 'WPX/NH-HAZE-2023/clean/10_GT.png',
        '51856f0d938a80799b2cb0031b9022af');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/011.JPG', '5a02dfe3c4c0125938299f47b1dd4964', 'WPX/NH-HAZE-2023/clean/11_GT.png',
        '3b00ffc8fc2d4aba140632f225145084');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/012.JPG', '672c953f00ca5dcc2101397daacbb4a4', 'WPX/NH-HAZE-2023/clean/12_GT.png',
        'bff4e8dbebe2aa8c81b9aa50e997399d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/013.JPG', 'f06e02a6e9ed52c65364928d56ea5c8d', 'WPX/NH-HAZE-2023/clean/13_GT.png',
        '0159e3e2aa7f1b54eb92938ac3643d4c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/014.JPG', '15b89753d2c806f760c2db573c873cc8', 'WPX/NH-HAZE-2023/clean/14_GT.png',
        '3883237af0b5307022521b9cd71c1bcf');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/015.JPG', '99a64294ef0371999164b16dd3ea07fd', 'WPX/NH-HAZE-2023/clean/15_GT.png',
        'ab072cbbbd152e9971d9e11b07cee28e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/016.JPG', 'bda91485486a19ca28f3f3f4c63b92be', 'WPX/NH-HAZE-2023/clean/16_GT.png',
        '0c18d8278c6ea3963eaa280783aa9a53');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/017.JPG', 'a1d32442564db7edae3391a855cb9c89', 'WPX/NH-HAZE-2023/clean/17_GT.png',
        'a43b334072bdf0cbeaf408c1e4bb15a5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/018.JPG', 'fb00c5546f15917d53fa14accb8689dc', 'WPX/NH-HAZE-2023/clean/18_GT.png',
        'adc139051bbf1fc03d877d1737382a11');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/019.JPG', '10c88792aad7e3ee9c7b17fcf46e1fca', 'WPX/NH-HAZE-2023/clean/19_GT.png',
        '252ab0d41603a96cb03fe4dca10a05e1');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/020.JPG', 'd27091dafdd1bf4b6d90b6d71f79cc78', 'WPX/NH-HAZE-2023/clean/20_GT.png',
        'fafa985c9f771555bd3d38251821143e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/021.JPG', '7467471cbb0a5512fe567973012a66bf', 'WPX/NH-HAZE-2023/clean/21_GT.png',
        '04246152b6baad0393cdca8c80d55d88');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/022.JPG', '3b7b6380e2993b834542654bbb6cad0e', 'WPX/NH-HAZE-2023/clean/22_GT.png',
        '6879bb52ab036223320b19ef742ef851');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/023.JPG', '34d0a8499bbeb0ae90cebbc5a0a92575', 'WPX/NH-HAZE-2023/clean/23_GT.png',
        '9f0010787f1c6c954068b5ff6ccca51b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/024.JPG', '3e0f128a819eb4260e166eac9d9fd077', 'WPX/NH-HAZE-2023/clean/24_GT.png',
        'e33a1e74c2a2deb7a606da7263d1921e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/025.JPG', '09e24178d365aa4758bc1db5bcb3c69e', 'WPX/NH-HAZE-2023/clean/25_GT.png',
        '6990dd267d6c7bdfbf9f4a19e77a1cef');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/026.JPG', '647fdf52acab1a74e20867544afec046', 'WPX/NH-HAZE-2023/clean/26_GT.png',
        'fbfcbd7e7ae0f4f04ebbdcde146ef9f5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/027.JPG', '215cb3221e6985938146ff9804c59385', 'WPX/NH-HAZE-2023/clean/27_GT.png',
        'c595453fed8339b5330e69fb1ad802b6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/028.JPG', 'cceb8f11c6a1d81cc3532dee36119720', 'WPX/NH-HAZE-2023/clean/28_GT.png',
        'bdcf66ba2d1e82d21e78cc7be02d1f7f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/029.JPG', 'bbd1ad2c97b25958aa823f456cb73784', 'WPX/NH-HAZE-2023/clean/29_GT.png',
        '04302e60cce557e60501c35280edc8e4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/030.JPG', '6cc52955f9aea5e373448d55237d1364', 'WPX/NH-HAZE-2023/clean/30_GT.png',
        '841b9586bfcd7529a24b9266584d7074');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/031.JPG', 'e701944811d603ea0f3a11732dfa7abf', 'WPX/NH-HAZE-2023/clean/31_GT.png',
        'a6a9a297cf4f5e4ee3b5c6a47e9614ca');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/032.JPG', 'c388c5702ee60dbe4a264ca6dc4cb237', 'WPX/NH-HAZE-2023/clean/32_GT.png',
        '35332fcc275d1887328a9b4a7fe1625d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/033.JPG', 'd1886865b8539a9b9346eb4b4f827071', 'WPX/NH-HAZE-2023/clean/33_GT.png',
        'fbb46e1711df1c9ffa3278c87b432f0e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/034.JPG', '5a5983dcdf8e44e958c80ad90b8a121c', 'WPX/NH-HAZE-2023/clean/34_GT.png',
        '0c74b570b06f9b31960701f8311a3f2d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/035.JPG', '05297bd83b14038e3117f8ccf9c953b3', 'WPX/NH-HAZE-2023/clean/35_GT.png',
        '5ed8074ae138b20f749b6e8665b58c9c');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/036.JPG', '78f944c804881b577bd38770d5703549', 'WPX/NH-HAZE-2023/clean/36_GT.png',
        'fe4aaf98271169bb619e42b7ecb42f03');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/037.JPG', '8836da89828f858a82ce64274175d88c', 'WPX/NH-HAZE-2023/clean/37_GT.png',
        'af9e0e834cdc3705b2e0809130136ad4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/038.JPG', 'be7161a1ec8685bbe3b8862eeb927439', 'WPX/NH-HAZE-2023/clean/38_GT.png',
        '19820d04653b99adad429083b8301d51');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/039.JPG', '22a682cd0aa23706219153be4cbb785e', 'WPX/NH-HAZE-2023/clean/39_GT.png',
        '3a24b7c592431c7063fca448ee13bc47');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/clean/040.JPG', '06b259acba188591a28ff129ecf1a965', 'WPX/NH-HAZE-2023/clean/40_GT.png',
        '1e1d7993dcce1f203233e897216c906d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/001.JPG', '118bb2d0cbb7af415e7a81f6fc46fccd', 'WPX/NH-HAZE-2023/hazy/01_hazy.png',
        '70d6d59cf6babc7283dd50963519c0c6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/002.JPG', '6749a8d3885928dfa74dfc6fa61541a2', 'WPX/NH-HAZE-2023/hazy/02_hazy.png',
        '601255f470e447bfbe3f8ea87ef061aa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/003.JPG', 'a4788196afea106ef89286a0da62c660', 'WPX/NH-HAZE-2023/hazy/03_hazy.png',
        '55e8c349b2d73317eefa5b98dae68266');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/004.JPG', 'cc51dca758ce6572f9a7c7a8e59f7b5b', 'WPX/NH-HAZE-2023/hazy/04_hazy.png',
        'e0d3da499d6257e22082c9187de5bee4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/005.JPG', 'c776b2b17e2e1eb99e8a885959c37378', 'WPX/NH-HAZE-2023/hazy/05_hazy.png',
        '4f5305dd78ba2236f9e2de3d05c901b8');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/006.JPG', 'ce56cdc301d765c3becdcc1758138f12', 'WPX/NH-HAZE-2023/hazy/06_hazy.png',
        '6c63612298d40b46d45a7e399eade50f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/007.JPG', '51ac098054fc20818232fa5ced43e601', 'WPX/NH-HAZE-2023/hazy/07_hazy.png',
        '4a3e8524d94d40ded8217a5ff45222e6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/008.JPG', '18c4310ef4a8aeeddca798fc5480a3f8', 'WPX/NH-HAZE-2023/hazy/08_hazy.png',
        '52c91af868bc96dd2bf3b91d1e352aae');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/009.JPG', '9f530931267e158e2c9f80feaad1cc5e', 'WPX/NH-HAZE-2023/hazy/09_hazy.png',
        '723d440310c4e75a4e3f7403e04648c4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/010.JPG', 'ad7459837803ebe208ca3f9a2c372fa5', 'WPX/NH-HAZE-2023/hazy/10_hazy.png',
        '205974a5f1e56e530eb74e56fdf696c6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/011.JPG', 'cb8b70ab5eb80cbd52f76ccc5c240516', 'WPX/NH-HAZE-2023/hazy/11_hazy.png',
        'dac32752fd4d2ee7b9796cadc8128819');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/012.JPG', 'a647a1065e4a8a192bd50b98fe597e5b', 'WPX/NH-HAZE-2023/hazy/12_hazy.png',
        '3bd32645ed7fff1cba40d2e927ee8203');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/013.JPG', '31f6c6b3051651e04ba2d1108ca33214', 'WPX/NH-HAZE-2023/hazy/13_hazy.png',
        '999773a7e3ce5b486b2ecaa50ac999db');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/014.JPG', '8b5c9ce2fec812cd3cb08df366a503bc', 'WPX/NH-HAZE-2023/hazy/14_hazy.png',
        '64ca5c17a1efb4c6e0dfcbded04d448f');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/015.JPG', '1739d5077f52dbb552e393f1351cf1e7', 'WPX/NH-HAZE-2023/hazy/15_hazy.png',
        '4dc69ffe4e7d4ea373c9bf5e98bf3bb4');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/016.JPG', '809110249a7c2c67cf54d20d9dc00158', 'WPX/NH-HAZE-2023/hazy/16_hazy.png',
        'e7aeaaad40e368ce71868c3830201431');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/017.JPG', '9ce1ff1296674b90f424870f9cbe9936', 'WPX/NH-HAZE-2023/hazy/17_hazy.png',
        'adf87c05bcd9b1be26ea4c21dafb930e');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/018.JPG', '00f5ed1e25c03ed622c90eb899209e74', 'WPX/NH-HAZE-2023/hazy/18_hazy.png',
        '2ec69e20415976ab8bb510cda685f771');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/019.JPG', '110218c0af5d4dbe02ed4656700bbc94', 'WPX/NH-HAZE-2023/hazy/19_hazy.png',
        '0f5455539297d91059e2908e6c1f69e9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/020.JPG', 'f630f3c1884029b601446d46df2d1663', 'WPX/NH-HAZE-2023/hazy/20_hazy.png',
        '86e323d6f1ac6cf155e8783d601521cd');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/021.JPG', '87597f354e75c2b4ec369ffcc550a585', 'WPX/NH-HAZE-2023/hazy/21_hazy.png',
        '2234b89bf7e892700d8dd054f03d2b75');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/022.JPG', 'c5dc91ec6f7095161033aa332a21b86c', 'WPX/NH-HAZE-2023/hazy/22_hazy.png',
        '0db58814869cf7c956eb61c4a384e657');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/023.JPG', 'cee1eb53e44e951cb40e32243e0f2e61', 'WPX/NH-HAZE-2023/hazy/23_hazy.png',
        '94a2cb7b5b27a24b2a7a9add47ed1131');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/024.JPG', '7e7bbeeb6e3d383c7b789c2e7009648f', 'WPX/NH-HAZE-2023/hazy/24_hazy.png',
        '58dd6309cd51fcb0f91c5306d33944e6');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/025.JPG', '1028d80071de1d58e8fad9b7106b1d20', 'WPX/NH-HAZE-2023/hazy/25_hazy.png',
        '32ec2cff7e4824af5c82202ae1a25c85');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/026.JPG', '1dc010fa38c3f1cff06230d6d62c43e7', 'WPX/NH-HAZE-2023/hazy/26_hazy.png',
        'a3913a91169ef038a8309a8eb53c4f8b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/027.JPG', 'b05393a879d0938137c040f6eaabf1b2', 'WPX/NH-HAZE-2023/hazy/27_hazy.png',
        '8d5d79fbff01b96eb6215bf94bf36baa');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/028.JPG', '99a7a0c4a9acc8cad95bf772263551c8', 'WPX/NH-HAZE-2023/hazy/28_hazy.png',
        '8acb949d5d4d18c26bcbf3071a9bc8f5');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/029.JPG', '4cdac2f718ab98072526481260655bf2', 'WPX/NH-HAZE-2023/hazy/29_hazy.png',
        '8c7956095ea50da60a3f6445abaa47f3');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/030.JPG', 'ae9e77aa4210abe8575b8450b1299e85', 'WPX/NH-HAZE-2023/hazy/30_hazy.png',
        'ce982850c5c030cc69f7f8fca86feb95');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/031.JPG', 'e4a57eef59503f0abede95b834ca5bf9', 'WPX/NH-HAZE-2023/hazy/31_hazy.png',
        'e0aeaa789a3f6ccb28072bfafa3fd040');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/032.JPG', '62947faa3ee1de31e34dd77a5c3fed73', 'WPX/NH-HAZE-2023/hazy/32_hazy.png',
        '4aee8b5a7445d482f6b876657bc2e166');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/033.JPG', '33d84061a6bdd35250be3aefcf70bc9d', 'WPX/NH-HAZE-2023/hazy/33_hazy.png',
        'bf52af84aeb14e824688c0c9609baa6b');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/034.JPG', 'fd2b9a3bad0991f16a0c4b76ec7ab929', 'WPX/NH-HAZE-2023/hazy/34_hazy.png',
        'dd14927b28df0c95e140cdeab6142f1d');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/035.JPG', 'f4de62eb80fe01ce2f64c4d9d59aa003', 'WPX/NH-HAZE-2023/hazy/35_hazy.png',
        '0411cf2208a3cc0062f74683d4c9ca79');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/036.JPG', 'fefcd72f33025c7d6c3b28094cc3cd4d', 'WPX/NH-HAZE-2023/hazy/36_hazy.png',
        'dae0447630a0f881f95d614f4b47bd33');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/037.JPG', '3d186e31bd44b365b8de9069cb43da19', 'WPX/NH-HAZE-2023/hazy/37_hazy.png',
        'cd3a39657b3e2b5ca32035085c1650a9');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/038.JPG', 'd93dfd3d1a13f567310bb507bafda01d', 'WPX/NH-HAZE-2023/hazy/38_hazy.png',
        '61730209c8d2855b5d618472cc9e2111');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/039.JPG', '9483fd150f386ab4344ddf842f87b4e4', 'WPX/NH-HAZE-2023/hazy/39_hazy.png',
        'fcdf629a9525a5d50438162991aa3bb7');
INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5)
VALUES ('NH-HAZE-2023/hazy/040.JPG', '1c607e508e9dd60251b2503916fa18d0', 'WPX/NH-HAZE-2023/hazy/40_hazy.png',
        'fad26d9292b2a1752d89bb384261b597');

SET FOREIGN_KEY_CHECKS = 1;

