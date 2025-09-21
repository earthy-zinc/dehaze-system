CREATE DATABASE IF NOT EXISTS `pei_crm`;
USE `pei_crm`;
-- ----------------------------
-- Table structure for crm_business
-- ----------------------------
DROP TABLE IF EXISTS `crm_business`;
CREATE TABLE `crm_business`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '商机主键',
    `name`                varchar(255)   NOT NULL DEFAULT '' COMMENT '商机名称',
    `customer_id`         bigint         NOT NULL DEFAULT 0 COMMENT '客户编号',
    `follow_up_status`    bit(1)         NOT NULL DEFAULT b'0' COMMENT '跟进状态',
    `contact_last_time`   datetime       NULL     DEFAULT NULL COMMENT '最后跟进时间',
    `contact_next_time`   datetime       NULL     DEFAULT NULL COMMENT '下次联系时间',
    `owner_user_id`       bigint         NOT NULL DEFAULT 0 COMMENT '负责人用户编号',
    `status_type_id`      bigint         NOT NULL DEFAULT 0 COMMENT '商机状态组编号',
    `status_id`           bigint         NOT NULL DEFAULT 0 COMMENT '商机状态编号',
    `end_status`          int            NOT NULL DEFAULT 0 COMMENT '结束状态。0: 未提交；10: 审批中；20: 审核通过；30: 审核不通过；40: 已取消',
    `end_remark`          varchar(512)   NULL     DEFAULT '' COMMENT '结束时的备注',
    `deal_time`           datetime       NULL     DEFAULT NULL COMMENT '预计成交日期',
    `total_product_price` decimal(10, 2) NULL     DEFAULT NULL COMMENT '产品总金额，单位：元',
    `discount_percent`    decimal(10, 2) NULL     DEFAULT NULL COMMENT '整单折扣，百分比',
    `total_price`         decimal(10, 2) NULL     DEFAULT NULL COMMENT '商机总金额，单位：元',
    `remark`              varchar(512)   NULL     DEFAULT '' COMMENT '备注',
    `creator`             varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE,
    INDEX `idx_status_type_id` (`status_type_id` ASC) USING BTREE,
    INDEX `idx_status_id` (`status_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 商机表';

-- ----------------------------
-- Table structure for crm_business_product
-- ----------------------------
DROP TABLE IF EXISTS `crm_business_product`;
CREATE TABLE `crm_business_product`
(
    `id`             bigint         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `business_id`    bigint         NOT NULL DEFAULT 0 COMMENT '商机编号',
    `product_id`     bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_price`  decimal(10, 2) NULL     DEFAULT NULL COMMENT '产品单价，单位：元',
    `business_price` decimal(10, 2) NULL     DEFAULT NULL COMMENT '商机价格，单位：元',
    `count`          decimal(10, 2) NULL     DEFAULT NULL COMMENT '数量',
    `total_price`    decimal(10, 2) NULL     DEFAULT NULL COMMENT '总计价格，单位：元',
    `creator`        varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_business_id` (`business_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 商机产品关联表';

-- ----------------------------
-- Table structure for crm_business_status
-- ----------------------------
DROP TABLE IF EXISTS `crm_business_status`;
CREATE TABLE `crm_business_status`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `type_id`     bigint       NOT NULL COMMENT '状态类型编号（关联 crm_business_status_type 表的 id 字段）',
    `name`        varchar(255) NOT NULL COMMENT '状态名',
    `percent`     int          NOT NULL COMMENT '赢单率，百分比',
    `sort`        int          NOT NULL COMMENT '排序',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_update_time` (`update_time` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 商机状态表';

-- ----------------------------
-- Table structure for crm_business_status_type
-- ----------------------------
DROP TABLE IF EXISTS `crm_business_status_type`;
CREATE TABLE `crm_business_status_type`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '状态类型名',
    `dept_ids`    text         NULL COMMENT '使用的部门编号列表',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 商机状态组表';

-- ----------------------------
-- Table structure for crm_clue
-- ----------------------------
DROP TABLE IF EXISTS `crm_clue`;
CREATE TABLE `crm_clue`
(
    `id`                   bigint       NOT NULL AUTO_INCREMENT COMMENT '线索主键',
    `name`                 varchar(255) NOT NULL DEFAULT '' COMMENT '线索名称',
    `follow_up_status`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '跟进状态',
    `contact_last_time`    datetime     NULL     DEFAULT NULL COMMENT '最后跟进时间',
    `contact_last_content` varchar(512) NULL     DEFAULT '' COMMENT '最后跟进内容',
    `contact_next_time`    datetime     NULL     DEFAULT NULL COMMENT '下次联系时间',
    `owner_user_id`        bigint       NOT NULL DEFAULT 0 COMMENT '负责人的用户编号',
    `transform_status`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '转化状态',
    `customer_id`          bigint       NOT NULL DEFAULT 0 COMMENT '客户编号',
    `mobile`               varchar(32)  NOT NULL DEFAULT '' COMMENT '手机号',
    `telephone`            varchar(32)  NOT NULL DEFAULT '' COMMENT '电话',
    `qq`                   varchar(32)  NOT NULL DEFAULT '' COMMENT 'QQ',
    `wechat`               varchar(64)  NOT NULL DEFAULT '' COMMENT '微信',
    `email`                varchar(64)  NOT NULL DEFAULT '' COMMENT '电子邮箱',
    `area_id`              int          NULL     DEFAULT NULL COMMENT '所在地编号',
    `detail_address`       varchar(255) NULL     DEFAULT '' COMMENT '详细地址',
    `industry_id`          int          NULL     DEFAULT NULL COMMENT '所属行业。参考字典crm_customer_industry',
    `level`                int          NULL     DEFAULT NULL COMMENT '客户等级。参考字典crm_customer_level',
    `source`               int          NULL     DEFAULT NULL COMMENT '客户来源。参考字典crm_customer_source',
    `remark`               varchar(512) NULL     DEFAULT '' COMMENT '备注',
    `creator`              varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 线索表';

-- ----------------------------
-- Table structure for crm_contact_business
-- ----------------------------
DROP TABLE IF EXISTS `crm_contact_business`;
CREATE TABLE `crm_contact_business`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `contact_id`  bigint      NOT NULL DEFAULT 0 COMMENT '联系人编号',
    `business_id` bigint      NOT NULL DEFAULT 0 COMMENT '商机编号',
    `creator`     varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_contact_id` (`contact_id` ASC) USING BTREE,
    INDEX `idx_business_id` (`business_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 联系人与商机关联表';

-- ----------------------------
-- Table structure for crm_contact
-- ----------------------------
DROP TABLE IF EXISTS `crm_contact`;
CREATE TABLE `crm_contact`
(
    `id`                   bigint       NOT NULL AUTO_INCREMENT COMMENT '联系人主键',
    `name`                 varchar(64)  NOT NULL DEFAULT '' COMMENT '联系人姓名',
    `customer_id`          bigint       NOT NULL DEFAULT 0 COMMENT '客户编号',
    `contact_last_time`    datetime     NULL     DEFAULT NULL COMMENT '最后跟进时间',
    `contact_last_content` varchar(512) NULL     DEFAULT '' COMMENT '最后跟进内容',
    `contact_next_time`    datetime     NULL     DEFAULT NULL COMMENT '下次联系时间',
    `owner_user_id`        bigint       NOT NULL DEFAULT 0 COMMENT '负责人用户编号',
    `mobile`               varchar(32)  NOT NULL DEFAULT '' COMMENT '手机号',
    `telephone`            varchar(32)  NOT NULL DEFAULT '' COMMENT '电话',
    `email`                varchar(64)  NOT NULL DEFAULT '' COMMENT '电子邮箱',
    `area_id`              int          NULL     DEFAULT NULL COMMENT '所在地编号',
    `detail_address`       varchar(255) NULL     DEFAULT '' COMMENT '详细地址',
    `sex`                  int          NULL     DEFAULT NULL COMMENT '性别。参考system模块的SexEnum',
    `master`               bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否关键决策人',
    `post`                 varchar(64)  NULL     DEFAULT '' COMMENT '职位',
    `parent_id`            bigint       NULL     DEFAULT NULL COMMENT '直属上级',
    `remark`               varchar(512) NULL     DEFAULT '' COMMENT '备注',
    `creator`              varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 联系人表';

-- ----------------------------
-- Table structure for crm_contract_config
-- ----------------------------
DROP TABLE IF EXISTS `crm_contract_config`;
CREATE TABLE `crm_contract_config`
(
    `id`             bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `notify_enabled` bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否开启提前提醒（0=关闭，1=开启）',
    `notify_days`    int         NOT NULL COMMENT '提前提醒天数',
    `creator`        varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 合同配置表';

-- ----------------------------
-- Table structure for crm_contract
-- ----------------------------
DROP TABLE IF EXISTS `crm_contract`;
CREATE TABLE `crm_contract`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '合同主键',
    `name`                varchar(255)   NOT NULL DEFAULT '' COMMENT '合同名称',
    `no`                  varchar(64)    NOT NULL DEFAULT '' COMMENT '合同编号',
    `customer_id`         bigint         NOT NULL DEFAULT 0 COMMENT '客户编号',
    `business_id`         bigint         NULL     DEFAULT NULL COMMENT '商机编号',
    `contact_last_time`   datetime       NULL     DEFAULT NULL COMMENT '最后跟进时间',
    `owner_user_id`       bigint         NOT NULL DEFAULT 0 COMMENT '负责人的用户编号',
    `process_instance_id` varchar(64)    NOT NULL DEFAULT '' COMMENT '工作流编号',
    `audit_status`        int            NOT NULL DEFAULT 0 COMMENT '审批状态。0: 未提交；10: 审批中；20: 审核通过；30: 审核不通过；40: 已取消',
    `order_date`          datetime       NULL     DEFAULT NULL COMMENT '下单日期',
    `start_time`          datetime       NULL     DEFAULT NULL COMMENT '开始时间',
    `end_time`            datetime       NULL     DEFAULT NULL COMMENT '结束时间',
    `total_product_price` decimal(10, 2) NULL     DEFAULT NULL COMMENT '产品总金额，单位：元',
    `discount_percent`    decimal(10, 2) NULL     DEFAULT NULL COMMENT '整单折扣',
    `total_price`         decimal(10, 2) NULL     DEFAULT NULL COMMENT '合同总金额，单位：分',
    `sign_contact_id`     bigint         NULL     DEFAULT NULL COMMENT '客户签约人',
    `sign_user_id`        bigint         NULL     DEFAULT NULL COMMENT '公司签约人',
    `remark`              varchar(512)   NULL     DEFAULT '' COMMENT '备注',
    `creator`             varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE,
    INDEX `idx_business_id` (`business_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE,
    INDEX `idx_process_instance_id` (`process_instance_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 合同表';

-- ----------------------------
-- Table structure for crm_contract_product
-- ----------------------------
DROP TABLE IF EXISTS `crm_contract_product`;
CREATE TABLE `crm_contract_product`
(
    `id`             bigint         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `contract_id`    bigint         NOT NULL DEFAULT 0 COMMENT '合同编号',
    `product_id`     bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_price`  decimal(10, 2) NULL     DEFAULT NULL COMMENT '产品单价，单位：元',
    `contract_price` decimal(10, 2) NULL     DEFAULT NULL COMMENT '合同价格，单位：元',
    `count`          decimal(10, 2) NULL     DEFAULT NULL COMMENT '数量',
    `total_price`    decimal(10, 2) NULL     DEFAULT NULL COMMENT '总计价格，单位：元',
    `creator`        varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_contract_id` (`contract_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 合同产品关联表';

-- ----------------------------
-- Table structure for crm_customer
-- ----------------------------
DROP TABLE IF EXISTS `crm_customer`;
CREATE TABLE `crm_customer`
(
    `id`                   bigint       NOT NULL AUTO_INCREMENT COMMENT '客户主键',
    `name`                 varchar(255) NOT NULL DEFAULT '' COMMENT '客户名称',
    `follow_up_status`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '跟进状态',
    `contact_last_time`    datetime     NULL     DEFAULT NULL COMMENT '最后跟进时间',
    `contact_last_content` varchar(512) NULL     DEFAULT '' COMMENT '最后跟进内容',
    `contact_next_time`    datetime     NULL     DEFAULT NULL COMMENT '下次联系时间',
    `owner_user_id`        bigint       NOT NULL DEFAULT 0 COMMENT '负责人的用户编号',
    `owner_time`           datetime     NULL     DEFAULT NULL COMMENT '成为负责人的时间',
    `lock_status`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '锁定状态',
    `deal_status`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '成交状态',
    `mobile`               varchar(32)  NOT NULL DEFAULT '' COMMENT '手机号',
    `telephone`            varchar(32)  NOT NULL DEFAULT '' COMMENT '电话',
    `qq`                   varchar(32)  NOT NULL DEFAULT '' COMMENT 'QQ',
    `wechat`               varchar(64)  NOT NULL DEFAULT '' COMMENT '微信',
    `email`                varchar(64)  NOT NULL DEFAULT '' COMMENT '电子邮箱',
    `area_id`              int          NULL     DEFAULT NULL COMMENT '所在地编号',
    `detail_address`       varchar(255) NULL     DEFAULT '' COMMENT '详细地址',
    `industry_id`          int          NULL     DEFAULT NULL COMMENT '所属行业。参考字典crm_customer_industry',
    `level`                int          NULL     DEFAULT NULL COMMENT '客户等级。参考字典crm_customer_level',
    `source`               int          NULL     DEFAULT NULL COMMENT '客户来源。参考字典crm_customer_source',
    `remark`               varchar(512) NULL     DEFAULT '' COMMENT '备注',
    `creator`              varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 客户表';

-- ----------------------------
-- Table structure for crm_customer_limit_config
-- ----------------------------
DROP TABLE IF EXISTS `crm_customer_limit_config`;
CREATE TABLE `crm_customer_limit_config`
(
    `id`                 bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `type`               int         NOT NULL DEFAULT 0 COMMENT '规则类型。0: 公海规则；1: 分配规则',
    `user_ids`           text        NULL     COMMENT '规则适用人群（逗号分隔）',
    `dept_ids`           text        NULL     COMMENT '规则适用部门（逗号分隔）',
    `max_count`          int         NOT NULL DEFAULT 0 COMMENT '数量上限',
    `deal_count_enabled` bit(1)      NOT NULL DEFAULT b'0' COMMENT '成交客户是否占有拥有客户数',
    `creator`            varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`        datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`        datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 客户限制配置表';

-- ----------------------------
-- Table structure for crm_customer_pool_config
-- ----------------------------
DROP TABLE IF EXISTS `crm_customer_pool_config`;
CREATE TABLE `crm_customer_pool_config`
(
    `id`                  bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `enabled`             bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否启用客户公海',
    `contact_expire_days` int         NULL     DEFAULT NULL COMMENT '未跟进放入公海天数',
    `deal_expire_days`    int         NULL     DEFAULT NULL COMMENT '未成交放入公海天数',
    `notify_enabled`      bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否开启提前提醒',
    `notify_days`         int         NULL     DEFAULT NULL COMMENT '提前提醒天数',
    `creator`             varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 客户公海配置表';

-- ----------------------------
-- Table structure for crm_follow_up_record
-- ----------------------------
DROP TABLE IF EXISTS `crm_follow_up_record`;
CREATE TABLE `crm_follow_up_record`
(
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `biz_type`     int         NOT NULL DEFAULT 0 COMMENT '数据类型。1: 线索；2: 客户；3: 联系人；4: 商机；5: 合同；6: 产品；7: 回款；8: 回款计划',
    `biz_id`       bigint      NOT NULL DEFAULT 0 COMMENT '数据编号',
    `type`         int         NOT NULL DEFAULT 0 COMMENT '跟进类型。参考字典crm_follow_up_type',
    `content`      text        NULL COMMENT '跟进内容',
    `next_time`    datetime    NULL COMMENT '下次联系时间',
    `pic_urls`     text        NULL COMMENT '图片URL数组（逗号分隔）',
    `file_urls`    text        NULL COMMENT '附件URL数组（逗号分隔）',
    `business_ids` text        NULL COMMENT '关联的商机编号数组（逗号分隔）',
    `contact_ids`  text        NULL COMMENT '关联的联系人编号数组（逗号分隔）',
    `creator`      varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_biz_type` (`biz_type` ASC) USING BTREE,
    INDEX `idx_biz_id` (`biz_id` ASC) USING BTREE,
    INDEX `idx_type` (`type` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 跟进记录表';

-- ----------------------------
-- Table structure for crm_permission
-- ----------------------------
DROP TABLE IF EXISTS `crm_permission`;
CREATE TABLE `crm_permission`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `biz_type`    int         NOT NULL DEFAULT 0 COMMENT '数据类型。1: 线索；2: 客户；3: 联系人；4: 商机；5: 合同；6: 产品；7: 回款；8: 回款计划',
    `biz_id`      bigint      NOT NULL DEFAULT 0 COMMENT '数据编号',
    `user_id`     bigint      NOT NULL DEFAULT 0 COMMENT '用户编号',
    `level`       int         NOT NULL DEFAULT 0 COMMENT '权限级别。1: 负责人；2: 只读；3: 读写',
    `creator`     varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_biz_type` (`biz_type` ASC) USING BTREE,
    INDEX `idx_biz_id` (`biz_id` ASC) USING BTREE,
    INDEX `idx_user_id` (`user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 数据权限表';

-- ----------------------------
-- Table structure for crm_product_category
-- ----------------------------
DROP TABLE IF EXISTS `crm_product_category`;
CREATE TABLE `crm_product_category`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '分类主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '分类名称',
    `parent_id`   bigint       NOT NULL DEFAULT 0 COMMENT '父级编号',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_parent_id` (`parent_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 产品分类表';

-- ----------------------------
-- Table structure for crm_product
-- ----------------------------
DROP TABLE IF EXISTS `crm_product`;
CREATE TABLE `crm_product`
(
    `id`            bigint         NOT NULL AUTO_INCREMENT COMMENT '产品主键',
    `name`          varchar(255)   NOT NULL DEFAULT '' COMMENT '产品名称',
    `no`            varchar(64)    NOT NULL DEFAULT '' COMMENT '产品编码',
    `unit`          int            NULL     DEFAULT NULL COMMENT '单位。参考字典crm_product_unit',
    `price`         decimal(10, 2) NULL     DEFAULT NULL COMMENT '价格，单位：元',
    `status`        int            NOT NULL DEFAULT 0 COMMENT '状态。0: 下架；1: 上架',
    `category_id`   bigint         NOT NULL DEFAULT 0 COMMENT '产品分类编号',
    `description`   varchar(512)   NULL     DEFAULT '' COMMENT '产品描述',
    `owner_user_id` bigint         NOT NULL DEFAULT 0 COMMENT '负责人的用户编号',
    `creator`       varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_category_id` (`category_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 产品表';

-- ----------------------------
-- Table structure for crm_receivable
-- ----------------------------
DROP TABLE IF EXISTS `crm_receivable`;
CREATE TABLE `crm_receivable`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '回款主键',
    `no`                  varchar(64)    NOT NULL DEFAULT '' COMMENT '回款编号',
    `plan_id`             bigint         NULL     DEFAULT NULL COMMENT '回款计划编号',
    `customer_id`         bigint         NOT NULL DEFAULT 0 COMMENT '客户编号',
    `contract_id`         bigint         NOT NULL DEFAULT 0 COMMENT '合同编号',
    `owner_user_id`       bigint         NOT NULL DEFAULT 0 COMMENT '负责人编号',
    `return_time`         datetime       NULL     DEFAULT NULL COMMENT '回款日期',
    `return_type`         int            NOT NULL DEFAULT 0 COMMENT '回款方式。1: 支票；2: 现金；3: 邮政汇款；4: 电汇；5: 网上转账；6: 支付宝；7: 微信支付；8: 其它',
    `price`               decimal(10, 2) NULL     DEFAULT NULL COMMENT '计划回款金额，单位：元',
    `remark`              varchar(512)   NULL     DEFAULT '' COMMENT '备注',
    `process_instance_id` varchar(64)    NOT NULL DEFAULT '' COMMENT '工作流编号',
    `audit_status`        int            NOT NULL DEFAULT 0 COMMENT '审批状态。0: 未提交；10: 审批中；20: 审核通过；30: 审核不通过；40: 已取消',
    `creator`             varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE,
    INDEX `idx_contract_id` (`contract_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE,
    INDEX `idx_process_instance_id` (`process_instance_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 回款表';

-- ----------------------------
-- Table structure for crm_receivable_plan
-- ----------------------------
DROP TABLE IF EXISTS `crm_receivable_plan`;
CREATE TABLE `crm_receivable_plan`
(
    `id`            bigint         NOT NULL AUTO_INCREMENT COMMENT '编号',
    `period`        int            NOT NULL DEFAULT 0 COMMENT '期数',
    `customer_id`   bigint         NOT NULL DEFAULT 0 COMMENT '客户编号',
    `contract_id`   bigint         NOT NULL DEFAULT 0 COMMENT '合同编号',
    `owner_user_id` bigint         NOT NULL DEFAULT 0 COMMENT '负责人编号',
    `return_time`   datetime       NULL     DEFAULT NULL COMMENT '计划回款日期',
    `return_type`   int            NOT NULL DEFAULT 0 COMMENT '计划回款类型。1: 支票；2: 现金；3: 邮政汇款；4: 电汇；5: 网上转账；6: 支付宝；7: 微信支付；8: 其它',
    `price`         decimal(10, 2) NULL     DEFAULT NULL COMMENT '计划回款金额，单位：元',
    `receivable_id` bigint         NULL     DEFAULT NULL COMMENT '回款编号',
    `remind_days`   int            NULL     DEFAULT NULL COMMENT '提前几天提醒',
    `remind_time`   datetime       NULL     DEFAULT NULL COMMENT '提醒日期',
    `remark`        varchar(512)   NULL     DEFAULT '' COMMENT '备注',
    `creator`       varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE,
    INDEX `idx_contract_id` (`contract_id` ASC) USING BTREE,
    INDEX `idx_owner_user_id` (`owner_user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'CRM 回款计划表';
