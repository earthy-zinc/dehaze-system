CREATE DATABASE IF NOT EXISTS `pei_bpm`;
USE `pei_bpm`;
-- ----------------------------
-- Table structure for bpm_category
-- ----------------------------
DROP TABLE IF EXISTS `bpm_category`;
CREATE TABLE `bpm_category`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '分类主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '分类名',
    `code`        varchar(64)  NOT NULL DEFAULT '' COMMENT '分类标志',
    `description` text         NULL COMMENT '描述',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '分类状态。0: 启用；1: 停用',
    `sort`        int          NOT NULL DEFAULT 0 COMMENT '排序值',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程分类表';

-- ----------------------------
-- Table structure for bpm_form
-- ----------------------------
DROP TABLE IF EXISTS `bpm_form`;
CREATE TABLE `bpm_form`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '表单主键',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '表单名',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '状态。0: 启用；1: 停用',
    `conf`        varchar(4096) NULL     DEFAULT NULL COMMENT '表单配置',
    `fields`      text          NULL     DEFAULT NULL COMMENT '表单项数组（JSON格式）',
    `remark`      varchar(512)  NULL     DEFAULT NULL COMMENT '备注',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程表单表';

-- ----------------------------
-- Table structure for bpm_process_definition_info
-- ----------------------------
DROP TABLE IF EXISTS `bpm_process_definition_info`;
CREATE TABLE `bpm_process_definition_info`
(
    `id`                             bigint        NOT NULL AUTO_INCREMENT COMMENT '流程定义信息主键',
    `process_definition_id`          varchar(64)   NOT NULL DEFAULT '' COMMENT '流程定义编号',
    `model_id`                       varchar(64)   NOT NULL DEFAULT '' COMMENT '模型编号',
    `model_type`                     int           NOT NULL DEFAULT 0 COMMENT '模型类型。1: 业务流程；2: 通用流程',
    `category`                       varchar(64)   NOT NULL DEFAULT '' COMMENT '流程分类编码',
    `icon`                           varchar(255)  NOT NULL DEFAULT '' COMMENT '图标',
    `description`                    varchar(1024) NULL     DEFAULT NULL COMMENT '描述',
    `form_type`                      int           NOT NULL DEFAULT 0 COMMENT '表单类型。1: 动态表单；2: 自定义表单',
    `form_id`                        bigint        NOT NULL DEFAULT 0 COMMENT '动态表单编号',
    `form_conf`                      varchar(4096) NULL     DEFAULT NULL COMMENT '表单配置',
    `form_fields`                    TEXT          NULL     DEFAULT NULL COMMENT '表单项数组（JSON格式）',
    `form_custom_create_path`        varchar(255)  NOT NULL DEFAULT '' COMMENT '自定义表单提交路径',
    `form_custom_view_path`          varchar(255)  NOT NULL DEFAULT '' COMMENT '自定义表单查看路径',
    `simple_model`                   TEXT          NULL     DEFAULT NULL COMMENT 'SIMPLE设计器模型数据（JSON格式）',
    `visible`                        bit(1)        NOT NULL DEFAULT b'1' COMMENT '是否可见',
    `sort`                           bigint        NOT NULL DEFAULT 0 COMMENT '排序值',
    `start_user_ids`                 text          NULL     DEFAULT NULL COMMENT '可发起用户编号数组（逗号分隔）',
    `start_dept_ids`                 text          NULL     DEFAULT NULL COMMENT '可发起部门编号数组（逗号分隔）',
    `manager_user_ids`               text          NULL     DEFAULT NULL COMMENT '可管理用户编号数组（逗号分隔）',
    `allow_cancel_running_process`   bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否允许撤销审批中的申请',
    `process_id_rule`                TEXT          NULL     DEFAULT NULL COMMENT '流程ID规则（JSON格式）',
    `auto_approval_type`             int           NULL     DEFAULT NULL COMMENT '自动去重类型。1: 按照申请人+标题+摘要重复校验；2: 不重复校验',
    `title_setting`                  TEXT          NULL     DEFAULT NULL COMMENT '标题设置（JSON格式）',
    `summary_setting`                TEXT          NULL     DEFAULT NULL COMMENT '摘要设置（JSON格式）',
    `process_before_trigger_setting` TEXT          NULL     DEFAULT NULL COMMENT '流程前置通知设置（JSON格式）',
    `process_after_trigger_setting`  TEXT          NULL     DEFAULT NULL COMMENT '流程后置通知设置（JSON格式）',
    `task_before_trigger_setting`    TEXT          NULL     DEFAULT NULL COMMENT '任务前置通知设置（JSON格式）',
    `task_after_trigger_setting`     TEXT          NULL     DEFAULT NULL COMMENT '任务后置通知设置（JSON格式）',
    `creator`                        varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`                    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                        varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`                    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                        bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_process_definition_id` (`process_definition_id` ASC) USING BTREE,
    INDEX `idx_model_id` (`model_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程定义信息表';

-- ----------------------------
-- Table structure for bpm_process_expression
-- ----------------------------
DROP TABLE IF EXISTS `bpm_process_expression`;
CREATE TABLE `bpm_process_expression`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '表达式主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '表达式名字',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '表达式状态。0: 启用；1: 停用',
    `expression`  text         NULL COMMENT '表达式',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程表达式表';

-- ----------------------------
-- Table structure for bpm_process_listener
-- ----------------------------
DROP TABLE IF EXISTS `bpm_process_listener`;
CREATE TABLE `bpm_process_listener`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '监听器主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '监听器名字',
    `status`      int          NOT NULL DEFAULT 0 COMMENT '状态',
    `type`        varchar(32)  NOT NULL DEFAULT '' COMMENT '监听类型',
    `event`       varchar(32)  NOT NULL DEFAULT '' COMMENT '监听事件',
    `value_type`  varchar(32)  NOT NULL DEFAULT '' COMMENT '值类型',
    `value`       varchar(255) NOT NULL DEFAULT '' COMMENT '值',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_type` (`type` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程监听器表';


-- ----------------------------
-- Table structure for bpm_user_group
-- ----------------------------
DROP TABLE IF EXISTS `bpm_user_group`;
CREATE TABLE `bpm_user_group`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '用户组主键',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '组名',
    `description` varchar(1024) NULL     DEFAULT NULL COMMENT '描述',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '状态。0: 启用；1: 停用',
    `user_ids`    text          NULL     DEFAULT NULL COMMENT '成员用户编号数组（逗号分隔）',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程用户组表';

-- ----------------------------
-- Table structure for bpm_oa_leave
-- ----------------------------
DROP TABLE IF EXISTS `bpm_oa_leave`;
CREATE TABLE `bpm_oa_leave`
(
    `id`                  bigint       NOT NULL AUTO_INCREMENT COMMENT '请假主键',
    `user_id`             bigint       NOT NULL DEFAULT 0 COMMENT '申请人用户编号',
    `type`                varchar(255) NOT NULL DEFAULT '' COMMENT '请假类型',
    `reason`              text         NULL COMMENT '原因',
    `start_time`          datetime     NOT NULL COMMENT '开始时间',
    `end_time`            datetime     NOT NULL COMMENT '结束时间',
    `day`                 bigint       NOT NULL DEFAULT 0 COMMENT '请假天数',
    `status`              int          NOT NULL DEFAULT 0 COMMENT '审批结果。-1: 未开始；1: 审批中；2: 审批通过；3: 审批不通过；4: 已取消；5: 已退回；7: 审批通过中；0: 待审批',
    `process_instance_id` varchar(64)  NOT NULL DEFAULT '' COMMENT '流程实例编号',
    `creator`             varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE,
    INDEX `idx_process_instance_id` (`process_instance_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'OA请假申请表';

-- ----------------------------
-- Table structure for bpm_process_instance_copy
-- ----------------------------
DROP TABLE IF EXISTS `bpm_process_instance_copy`;
CREATE TABLE `bpm_process_instance_copy`
(
    `id`                    bigint       NOT NULL AUTO_INCREMENT COMMENT '抄送主键',
    `start_user_id`         bigint       NOT NULL DEFAULT 0 COMMENT '发起人Id',
    `process_instance_name` varchar(255) NOT NULL DEFAULT '' COMMENT '流程实例名称',
    `process_instance_id`   varchar(64)  NOT NULL DEFAULT '' COMMENT '流程实例编号',
    `process_definition_id` varchar(64)  NOT NULL DEFAULT '' COMMENT '流程定义编号',
    `category`              varchar(64)  NOT NULL DEFAULT '' COMMENT '流程分类',
    `activity_id`           varchar(64)  NOT NULL DEFAULT '' COMMENT '流程活动编号',
    `activity_name`         varchar(255) NOT NULL DEFAULT '' COMMENT '流程活动名字',
    `task_id`               varchar(64)  NOT NULL DEFAULT '' COMMENT '任务编号',
    `user_id`               bigint       NOT NULL DEFAULT 0 COMMENT '被抄送的用户编号',
    `reason`                text         NULL COMMENT '抄送意见',
    `creator`               varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`           datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`               varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`           datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`               bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_process_instance_id` (`process_instance_id` ASC) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '流程抄送表';
