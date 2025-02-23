/*
* Pei 图像去雾系统数据库(MySQL8.x)
* @author earthyzinc
*/

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
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '部门表'
  ROW_FORMAT = DYNAMIC;


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
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典数据表'
  ROW_FORMAT = DYNAMIC;

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
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典类型表'
  ROW_FORMAT = DYNAMIC;

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
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  ROW_FORMAT = DYNAMIC COMMENT ='菜单管理';

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
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '角色表'
  ROW_FORMAT = DYNAMIC;

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
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '用户信息表'
  ROW_FORMAT = DYNAMIC;


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


DROP TABLE IF EXISTS `sys_dataset`;
CREATE TABLE `sys_dataset`
(
    `id`          bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '数据集ID',
    `parent_id`   bigint                                                         NOT NULL DEFAULT 0 COMMENT '父数据集ID',
    `type`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci   NOT NULL DEFAULT '' COMMENT '数据集类型',
    `name`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci   NOT NULL DEFAULT '' COMMENT '数据集名称',
    `img` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '数据集样例图片',
    `description` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL     DEFAULT '' COMMENT '数据集描述',
    `path`        varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NOT NULL DEFAULT '' COMMENT '存储位置',
    `size`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL     DEFAULT '' COMMENT '占用空间大小',
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


DROP TABLE IF EXISTS `sys_algorithm`;
CREATE TABLE `sys_algorithm`
(
    `id`     bigint      NOT NULL AUTO_INCREMENT COMMENT '模型id',
    `parent_id`   bigint           DEFAULT 0 COMMENT '模型的父id',
    `type`        varchar(100)     DEFAULT '' COMMENT '模型类型',
    `name`   varchar(64) NOT NULL COMMENT '模型名称',
    `img`    TEXT         DEFAULT NULL COMMENT '模型图片',
    `path`        varchar(255)     DEFAULT '' COMMENT '模型存储路径',
    `size`        varchar(100)     DEFAULT NULL COMMENT '模型大小',
    `params` varchar(255) DEFAULT NULL COMMENT '模型参数',
    `flops`  varchar(255) DEFAULT NULL COMMENT '模型浮点运算次数',
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


DROP TABLE IF EXISTS `sys_file`;
CREATE TABLE `sys_file`
(
    `id`          int             NOT NULL AUTO_INCREMENT COMMENT '文件id',
    `type`        varchar(100)             DEFAULT NULL COMMENT '文件类型',
    `url`         TEXT                     DEFAULT NULL COMMENT '文件url',
    `name`        varchar(100)    NOT NULL COMMENT '文件原始名',
    `object_name` varchar(100)    NOT NULL COMMENT '文件存储名',
    `size`        varchar(100)    NOT NULL DEFAULT '0' COMMENT '文件大小',
    `path`        varchar(255)    NOT NULL COMMENT '文件路径',
    `md5` char(32) UNIQUE NOT NULL COMMENT '文件的MD5值，用于比对文件是否相同',
    `create_time` datetime        NOT NULL COMMENT '创建时间',
    `update_time` datetime                 DEFAULT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `md5_key` (`md5` ASC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='文件表';

DROP TABLE IF EXISTS `sys_dataset_item`;
CREATE TABLE `sys_dataset_item`
(
    `id`         bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `dataset_id` bigint      NOT NULL COMMENT '所属数据集id',
    `name`       varchar(64) NULL COMMENT '数据项名称',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_dataset_id` (`dataset_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据集与数据项关联表';

DROP TABLE IF EXISTS `sys_item_file`;
CREATE TABLE `sys_item_file`
(
    `id`                bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `item_id`           bigint      NOT NULL COMMENT '所属数据项id',
    `file_id`           bigint      NOT NULL COMMENT '文件id',
    `thumbnail_file_id` bigint       DEFAULT NULL COMMENT '缩略图文件id',
    `type`              varchar(64) NOT NULL COMMENT '图片类型（清晰图、雾霾图、分割图等）',
    `description`       varchar(255) DEFAULT NULL COMMENT '描述',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_item_id_file_id` (`item_id`, `file_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据项图片关联表';

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
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_origin_md5` (`origin_md5`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='WPX文件表';


DROP TABLE IF EXISTS `sys_pred_log`;
CREATE TABLE `sys_pred_log`
(
    `id`             bigint   NOT NULL AUTO_INCREMENT COMMENT 'id',
    `algorithm_id`   bigint   NOT NULL COMMENT '算法id',
    `origin_file_id` bigint            DEFAULT NULL COMMENT '原始图像文件id（有雾图像）',
    `origin_md5`     char(32) NOT NULL COMMENT '原始图像md5值',
    `origin_url`     TEXT     NOT NULL COMMENT '原始图像url',
    `pred_file_id`   bigint            DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`       char(32) NOT NULL COMMENT '预测图像md5值',
    `pred_url`       TEXT     NOT NULL COMMENT '预测图像url',
    `time`           int               DEFAULT 0 COMMENT '推理时间（秒）',
    `create_time`    datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint   NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint   NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    KEY `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    KEY `idx_origin_md5` (`origin_md5`) USING BTREE,
    KEY `idx_pred_md5` (`pred_md5`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='模型预测日志表';

DROP TABLE IF EXISTS `sys_eval_log`;
CREATE TABLE `sys_eval_log`
(
    `id`           bigint   NOT NULL AUTO_INCREMENT COMMENT 'id',
    `algorithm_id` bigint   NOT NULL COMMENT '算法id',
    `pred_file_id` bigint            DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`     char(32) NOT NULL COMMENT '预测图像md5值',
    `pred_url`     TEXT     NOT NULL COMMENT '预测图像url',
    `gt_file_id`   bigint            DEFAULT NULL COMMENT '真值图像文件id',
    `gt_md5`       char(32) NOT NULL COMMENT '真值图像md5值',
    `gt_url`       TEXT     NOT NULL COMMENT '真值图像url',
    `time`         int               DEFAULT 0 COMMENT '评估时间（秒）',
    `result`       json COMMENT '预测结果',
    `create_time`  datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint   NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint   NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    KEY `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    KEY `idx_pred_md5` (`pred_md5`) USING BTREE,
    KEY `idx_gt_md5` (`gt_md5`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='模型预测日志表';

SET FOREIGN_KEY_CHECKS = 1;
