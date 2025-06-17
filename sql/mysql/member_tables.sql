CREATE DATABASE IF NOT EXISTS `pei_member`;
USE `pei_member`;


-- ----------------------------
-- Table structure for member_address
-- ----------------------------
DROP TABLE IF EXISTS `member_address`;
CREATE TABLE `member_address`
(
    `id`             bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`        bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `name`           varchar(255) NOT NULL DEFAULT '' COMMENT '收件人名称',
    `mobile`         varchar(20)  NOT NULL DEFAULT '' COMMENT '手机号',
    `area_id`        bigint       NOT NULL DEFAULT 0 COMMENT '地区编号',
    `detail_address` varchar(512) NOT NULL DEFAULT '' COMMENT '收件详细地址',
    `default_status` bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否默认: 0-否, 1-是',
    `creator`        varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='用户收件地址表';

-- ----------------------------
-- Table structure for member_config
-- ----------------------------
DROP TABLE IF EXISTS `member_config`;
CREATE TABLE `member_config`
(
    `id`                            bigint      NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    `point_trade_deduct_enable`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '积分抵扣开关: 0-关闭, 1-开启',
    `point_trade_deduct_unit_price` int         NOT NULL DEFAULT 0 COMMENT '积分抵扣单位价格(分): 1 积分抵扣多少分',
    `point_trade_deduct_max_price`  int         NOT NULL DEFAULT 0 COMMENT '积分抵扣最大值',
    `point_trade_give_point`        int         NOT NULL DEFAULT 0 COMMENT '1 元赠送多少分',
    `creator`                       varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`                   datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                       varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`                   datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                       bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='会员配置表';

-- ----------------------------
-- Table structure for member_group
-- ----------------------------
DROP TABLE IF EXISTS `member_group`;
CREATE TABLE `member_group`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '名称',
    `remark`      varchar(512) NOT NULL DEFAULT '' COMMENT '备注',
    `status`      tinyint      NOT NULL DEFAULT 0 COMMENT '状态: 0-禁用, 1-启用',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_create_time` (`create_time`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='用户分组表';

-- ----------------------------
-- Table structure for member_level
-- ----------------------------
DROP TABLE IF EXISTS `member_level`;
CREATE TABLE `member_level`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `name`             varchar(255) NOT NULL DEFAULT '' COMMENT '等级名称',
    `level`            int          NOT NULL DEFAULT 0 COMMENT '等级',
    `experience`       int          NOT NULL DEFAULT 0 COMMENT '升级经验',
    `discount_percent` int          NOT NULL DEFAULT 0 COMMENT '享受折扣百分比',
    `icon`             varchar(512) NOT NULL DEFAULT '' COMMENT '等级图标',
    `background_url`   varchar(512) NOT NULL DEFAULT '' COMMENT '等级背景图',
    `status`           tinyint      NOT NULL DEFAULT 0 COMMENT '状态: 0-禁用, 1-启用',
    `creator`          varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='会员等级配置表';
-- ----------------------------
-- Table structure for member_level_record
-- ----------------------------
DROP TABLE IF EXISTS `member_level_record`;
CREATE TABLE `member_level_record`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`          bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `level_id`         bigint       NOT NULL DEFAULT 0 COMMENT '等级编号',
    `level`            int          NOT NULL DEFAULT 0 COMMENT '会员等级',
    `discount_percent` int          NOT NULL DEFAULT 0 COMMENT '享受折扣百分比',
    `experience`       int          NOT NULL DEFAULT 0 COMMENT '升级经验',
    `user_experience`  int          NOT NULL DEFAULT 0 COMMENT '会员此时的经验',
    `remark`           varchar(512) NOT NULL DEFAULT '' COMMENT '备注',
    `description`      varchar(512) NOT NULL DEFAULT '' COMMENT '描述',
    `creator`          varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='会员等级变更记录表';
-- ----------------------------
-- Table structure for member_experience_record
-- ----------------------------
DROP TABLE IF EXISTS `member_experience_record`;
CREATE TABLE `member_experience_record`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`          bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `biz_type`         tinyint      NOT NULL DEFAULT 0 COMMENT '业务类型: 1-订单消费, 2-签到, 3-活动奖励',
    `biz_id`           varchar(64)  NOT NULL DEFAULT '' COMMENT '业务编号',
    `title`            varchar(255) NOT NULL DEFAULT '' COMMENT '标题',
    `description`      varchar(512) NOT NULL DEFAULT '' COMMENT '描述',
    `experience`       int          NOT NULL DEFAULT 0 COMMENT '获得经验',
    `total_experience` int          NOT NULL DEFAULT 0 COMMENT '变更后的总经验',
    `creator`          varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='会员经验记录表';

-- ----------------------------
-- Table structure for member_point_record
-- ----------------------------
DROP TABLE IF EXISTS `member_point_record`;
CREATE TABLE `member_point_record`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    `user_id`     bigint       NOT NULL DEFAULT 0 COMMENT '用户编号',
    `biz_id`      varchar(64)  NOT NULL DEFAULT '' COMMENT '业务编码',
    `biz_type`    tinyint      NOT NULL DEFAULT 0 COMMENT '业务类型: 1-签到, 2-订单消费, 3-活动奖励',
    `title`       varchar(255) NOT NULL DEFAULT '' COMMENT '积分标题',
    `description` varchar(512) NOT NULL DEFAULT '' COMMENT '积分描述',
    `point`       int          NOT NULL DEFAULT 0 COMMENT '变动积分: 正数表示获得积分, 负数表示消耗积分',
    `total_point` int          NOT NULL DEFAULT 0 COMMENT '变动后的总积分',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='用户积分记录表';


-- ----------------------------
-- Table structure for member_sign_in_config
-- ----------------------------
DROP TABLE IF EXISTS `member_sign_in_config`;
CREATE TABLE `member_sign_in_config`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '规则自增主键',
    `day`         int         NOT NULL DEFAULT 0 COMMENT '签到第 x 天',
    `point`       int         NOT NULL DEFAULT 0 COMMENT '奖励积分',
    `experience`  int         NOT NULL DEFAULT 0 COMMENT '奖励经验',
    `status`      tinyint     NOT NULL DEFAULT 0 COMMENT '状态: 0-禁用, 1-启用',
    `creator`     varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='用户签到规则配置表';

-- ----------------------------
-- Table structure for member_sign_in_record
-- ----------------------------
DROP TABLE IF EXISTS `member_sign_in_record`;
CREATE TABLE `member_sign_in_record`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`     bigint      NOT NULL DEFAULT 0 COMMENT '签到用户编号',
    `day`         int         NOT NULL DEFAULT 0 COMMENT '第几天签到',
    `point`       int         NOT NULL DEFAULT 0 COMMENT '签到获得的积分',
    `experience`  int         NOT NULL DEFAULT 0 COMMENT '签到获得的经验',
    `creator`     varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='用户签到记录表';

-- ----------------------------
-- Table structure for member_tag
-- ----------------------------
DROP TABLE IF EXISTS `member_tag`;
CREATE TABLE `member_tag`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '标签编号',
    `name`        varchar(255) NOT NULL COMMENT '标签名称',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员标签表';

-- ----------------------------
-- Table structure for member_user
-- ----------------------------
DROP TABLE IF EXISTS `member_user`;
CREATE TABLE `member_user`
(
    `id`                bigint        NOT NULL AUTO_INCREMENT COMMENT '用户ID',
    `mobile`            varchar(32)   NOT NULL COMMENT '手机号',
    `password`          varchar(100)  NOT NULL COMMENT '加密后的密码',
    `status`            tinyint       NOT NULL COMMENT '账号状态: 0-开启, 1-禁用',
    `register_ip`       varchar(50)   NOT NULL COMMENT '注册IP',
    `register_terminal` tinyint       NOT NULL COMMENT '注册终端: 1-PC, 2-移动端, 3-小程序',
    `login_ip`          varchar(50)   NOT NULL COMMENT '最后登录IP',
    `login_date`        datetime      NOT NULL COMMENT '最后登录时间',
    `nickname`          varchar(64)   NOT NULL COMMENT '用户昵称',
    `avatar`            varchar(1024) NOT NULL COMMENT '用户头像',
    `name`              varchar(64)   NOT NULL COMMENT '真实名字',
    `sex`               tinyint       NOT NULL COMMENT '性别: 0-男, 1-女, 2-未知',
    `birthday`          datetime      NOT NULL COMMENT '出生日期',
    `area_id`           int           NOT NULL COMMENT '所在地区域编号',
    `mark`              varchar(255)  NOT NULL COMMENT '用户备注',
    `point`             int           NOT NULL COMMENT '积分',
    `tag_ids`           text          NULL COMMENT '会员标签列表',
    `level_id`          bigint        NOT NULL COMMENT '会员级别编号',
    `experience`        int           NOT NULL COMMENT '会员经验',
    `group_id`          bigint        NOT NULL COMMENT '用户分组编号',
    `creator`           varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`         bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE KEY `uk_mobile` (`mobile`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员用户表';
