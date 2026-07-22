-- 数据库由 Docker MYSQL_DATABASE 环境变量或运维脚本自动创建
-- 此文件仅负责建表，不包含 CREATE DATABASE / USE 语句

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

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
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典数据表'
  ROW_FORMAT = DYNAMIC;

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
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `type_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '字典类型表'
  ROW_FORMAT = DYNAMIC;

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
    `status`      tinyint(1)                                                   NOT NULL DEFAULT '1' COMMENT '状态(1-启用;0-禁用)',
    `sort`        int                                                                   DEFAULT '0' COMMENT '排序',
    `icon`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci          DEFAULT '' COMMENT '菜单图标',
    `redirect`    varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci         DEFAULT NULL COMMENT '跳转路径',
    `create_time` datetime                                                              DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                              DEFAULT NULL COMMENT '更新时间',
    `always_show` tinyint                                                               DEFAULT NULL COMMENT '【目录】只有一个子路由是否始终显示(1:是 0:否)',
    `keep_alive`  tinyint                                                               DEFAULT NULL COMMENT '【菜单】是否开启页面缓存(1:是 0:否)',
    `create_by`   bigint                                                               DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                               DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_general_ci
  ROW_FORMAT = DYNAMIC COMMENT ='菜单管理';

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

DROP TABLE IF EXISTS `sys_role_menu`;
CREATE TABLE `sys_role_menu`
(
    `role_id` bigint NOT NULL COMMENT '角色ID',
    `menu_id` bigint NOT NULL COMMENT '菜单ID'
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '角色和菜单关联表'
  ROW_FORMAT = DYNAMIC;

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
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `login_name` (`username` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '用户信息表'
  ROW_FORMAT = DYNAMIC;

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
    `img`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci          NULL     DEFAULT NULL COMMENT '数据集样例图片',
    `description` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL     DEFAULT '' COMMENT '数据集描述',
    `path`        varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NOT NULL DEFAULT '' COMMENT '存储位置',
    `size`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci  NULL     DEFAULT '' COMMENT '占用空间大小',
    `status`      tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用；0:禁用)',
    `usage_count` BIGINT                                                         NOT NULL DEFAULT 0 COMMENT '使用次数',
    `deleted`     tinyint                                                        NULL     DEFAULT 0 COMMENT '逻辑删除标识(1:已删除;0:未删除)',
    `create_time` datetime                                                       NULL     DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime                                                       NULL     DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint                                                         NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                         NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_parent_id` (`parent_id`) USING BTREE,
    INDEX `idx_name` (`name`) USING BTREE,
    INDEX `idx_parent_name` (`parent_id`, `name`) USING BTREE,
    INDEX `idx_deleted` (`deleted`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = '数据集表'
  ROW_FORMAT = DYNAMIC;

DROP TABLE IF EXISTS `sys_algorithm`;
CREATE TABLE `sys_algorithm`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '模型id',
    `parent_id`   bigint           DEFAULT 0 COMMENT '模型的父id',
    `type`        varchar(100)     DEFAULT '' COMMENT '模型类型',
    `version`     varchar(50)      DEFAULT NULL COMMENT '算法版本号',
    `name`        varchar(64) NOT NULL COMMENT '模型名称',
    `img`         TEXT             DEFAULT NULL COMMENT '模型图片',
    `path`        varchar(255)     DEFAULT '' COMMENT '模型存储路径',
    `size`        varchar(100)     DEFAULT NULL COMMENT '模型大小',
    `params`      varchar(255)     DEFAULT NULL COMMENT '模型参数',
    `flops`       varchar(255)     DEFAULT NULL COMMENT '模型浮点运算次数',
    `import_path` varchar(255)     DEFAULT NULL COMMENT '模型代码导入路径',
    `description` varchar(2048)    DEFAULT NULL COMMENT '针对该模型的详细描述',
    `status`      tinyint(1)       DEFAULT 1 COMMENT '状态(1:启用；0:禁用)',
    `audit_by`    bigint           DEFAULT NULL COMMENT '审核人ID',
    `audit_time`  datetime         DEFAULT NULL COMMENT '审核时间',
    `audit_remark` varchar(500)    DEFAULT NULL COMMENT '审核备注',
    `create_time` datetime         DEFAULT NULL COMMENT '创建时间',
    `update_time` datetime         DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint      NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint      NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法模型表';

DROP TABLE IF EXISTS `sys_algorithm_version`;
CREATE TABLE `sys_algorithm_version`
(
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `algorithm_id` bigint      NOT NULL COMMENT '关联算法ID',
    `version`      varchar(50) NOT NULL COMMENT '版本号',
    `change_log`   TEXT        NULL COMMENT '变更日志',
    `status`       int         NULL COMMENT '该版本时的状态',
    `config_json`  TEXT        NULL COMMENT '该版本时的配置JSON',
    `model_file_id` bigint     NULL COMMENT '模型文件ID',
    `is_active`    tinyint(1)  NULL DEFAULT 0 COMMENT '是否当前活跃版本',
    `create_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint      NULL COMMENT '创建人ID',
    `update_by`    bigint      NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_algo_version` (`algorithm_id`, `version`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法版本历史表';

DROP TABLE IF EXISTS `sys_input_history`;
CREATE TABLE `sys_input_history`
(
    `id`                    bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`               bigint       NOT NULL COMMENT '用户ID',
    `original_image_url`    varchar(500) NULL COMMENT '原始图片URL',
    `original_thumbnail_url` varchar(500) NULL COMMENT '原始缩略图URL',
    `result_image_url`      varchar(500) NULL COMMENT '处理结果图片URL',
    `result_thumbnail_url`  varchar(500) NULL COMMENT '结果缩略图URL',
    `algorithm_id`          bigint       NULL COMMENT '算法ID',
    `algorithm_name`        varchar(100) NULL COMMENT '算法名称（冗余）',
    `algorithm_params`      TEXT         NULL COMMENT '算法参数（JSON）',
    `processing_time`       int          NULL COMMENT '处理耗时（毫秒）',
    `status`                tinyint      NULL DEFAULT 3 COMMENT '处理状态（1=成功，2=失败，3=处理中）',
    `input_source`          varchar(20)  NULL COMMENT '图片来源（upload/camera/sample）',
    `is_favorite`           tinyint(1)   NULL DEFAULT 0 COMMENT '是否收藏',
    `sync_status`           tinyint      NULL DEFAULT 0 COMMENT '同步状态（0=未同步，1=已同步）',
    `create_time`           datetime     NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`           datetime     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`             bigint       NULL COMMENT '创建人ID',
    `update_by`             bigint       NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_time` (`user_id`, `create_time` DESC) USING BTREE,
    INDEX `idx_user_favorite` (`user_id`, `is_favorite`, `create_time` DESC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='图像输入历史记录表';

DROP TABLE IF EXISTS `sys_algorithm_favorite`;
CREATE TABLE `sys_algorithm_favorite`
(
    `id`            bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `user_id`       bigint      NOT NULL COMMENT '用户ID',
    `algorithm_id`  bigint      NOT NULL COMMENT '算法ID',
    `create_time`   datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_algorithm` (`user_id`, `algorithm_id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法收藏表';

DROP TABLE IF EXISTS `sys_file`;
CREATE TABLE `sys_file`
(
    `id`          int             NOT NULL AUTO_INCREMENT COMMENT '文件id',
    `type`        varchar(100)             DEFAULT NULL COMMENT '文件类型',
    `url`         TEXT                     DEFAULT NULL COMMENT '文件url',
    `name`        varchar(100)    NOT NULL COMMENT '文件原始名',
    `object_name` varchar(100)    NOT NULL COMMENT '文件存储名',
    `size`        varchar(100)    NOT NULL DEFAULT '0' COMMENT '文件大小（格式化显示）',
    `size_bytes`  bigint                   DEFAULT NULL COMMENT '文件大小（原始字节数）',
    `path`        varchar(255)    NOT NULL COMMENT '文件路径',
    `md5`         char(32) UNIQUE NOT NULL COMMENT '文件的MD5值，用于比对文件是否相同',
    `create_time` datetime        NOT NULL COMMENT '创建时间',
    `update_time` datetime                 DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint                   DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                   DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `md5_key` (`md5` ASC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='文件表';

DROP TABLE IF EXISTS `sys_dataset_item`;
CREATE TABLE `sys_dataset_item`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `dataset_id`  bigint      NOT NULL COMMENT '所属数据集id',
    `name`        varchar(64) NULL COMMENT '数据项名称',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
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
    `type`              varchar(64) NOT NULL COMMENT '图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)',
    `description`       varchar(255) DEFAULT NULL COMMENT '描述',
    `scene_type`        varchar(64)  DEFAULT NULL COMMENT '场景类型',
    `haze_level`        varchar(32)  DEFAULT NULL COMMENT '雾霾程度标识，支持多种规范：light/medium/heavy（人工分级），beta=0.5（β参数），A=0.8,beta=0.2（大气光A+β双参数），空值表示未标注或无雾',
    `width`             int          DEFAULT NULL COMMENT '图片宽度',
    `height`            int          DEFAULT NULL COMMENT '图片高度',
    `usage_count`       bigint       DEFAULT 0 COMMENT '使用次数',
    `create_time`       datetime     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_item_id_file_id` (`item_id`, `file_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据项图片关联表';

DROP TABLE IF EXISTS `sys_task`;
CREATE TABLE `sys_task`
(
    `id`              BIGINT      NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `task_id`         VARCHAR(64) NOT NULL COMMENT '任务ID（UUID）',
    `task_type`       VARCHAR(32) NOT NULL COMMENT '任务类型',
    `status`          VARCHAR(32) NOT NULL DEFAULT 'PENDING' COMMENT '任务状态',
    `progress`        INT                  DEFAULT 0 COMMENT '任务进度（百分比）',
    `total_files`     INT                  DEFAULT 0 COMMENT '总文件数',
    `processed_files` INT                  DEFAULT 0 COMMENT '已处理文件数',
    `params`          TEXT COMMENT '任务参数（JSON）',
    `result`          TEXT COMMENT '任务结果（下载链接）',
    `error_message`   TEXT COMMENT '错误信息',
    `create_by`       BIGINT COMMENT '创建人ID',
    `update_by`       BIGINT COMMENT '修改人ID',
    `create_time`     DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     DATETIME    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `started_at`      DATETIME COMMENT '开始时间',
    `completed_at`    DATETIME COMMENT '完成时间',
    `expires_at`      DATETIME COMMENT '过期时间',
    `idempotency_key` VARCHAR(64) COMMENT '客户端幂等键（HTTP Idempotency-Key 头）',
    `retry_count`     INT         NOT NULL DEFAULT 0 COMMENT 'MQ 重试次数',
    `worker_id`       VARCHAR(64) COMMENT '执行 Worker 标识',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `idx_task_id` (`task_id`) USING BTREE,
    UNIQUE INDEX `idx_idempotency_key` (`idempotency_key`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_create_by` (`create_by`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='系统任务表';

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
    `origin_md5`     char(32) NULL DEFAULT NULL COMMENT '原始图像md5值',
    `origin_url`     TEXT     NULL DEFAULT NULL COMMENT '原始图像url',
    `pred_file_id`   bigint            DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`       char(32) NULL DEFAULT NULL COMMENT '预测图像md5值',
    `pred_url`       TEXT     NULL DEFAULT NULL COMMENT '预测图像url',
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
    `pred_md5`     char(32) NULL DEFAULT NULL COMMENT '预测图像md5值',
    `pred_url`     TEXT     NULL DEFAULT NULL COMMENT '预测图像url',
    `gt_file_id`   bigint            DEFAULT NULL COMMENT '真值图像文件id',
    `gt_md5`       char(32) NULL DEFAULT NULL COMMENT '真值图像md5值',
    `gt_url`       TEXT     NULL DEFAULT NULL COMMENT '真值图像url',
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

