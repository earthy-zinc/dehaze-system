CREATE DATABASE IF NOT EXISTS `pei_pay`;
USE `pei_pay`;

-- ----------------------------
-- Table structure for pay_app
-- ----------------------------
DROP TABLE IF EXISTS `pay_app`;
CREATE TABLE `pay_app`
(
    `id`                  bigint        NOT NULL AUTO_INCREMENT COMMENT '应用编号',
    `app_key`             varchar(64)   NOT NULL DEFAULT '' COMMENT '应用标识',
    `name`                varchar(64)   NOT NULL COMMENT '应用名',
    `status`              int           NOT NULL DEFAULT 0 COMMENT '状态',
    `remark`              varchar(512)  NULL     DEFAULT '' COMMENT '备注',
    `order_notify_url`    varchar(1024) NULL     DEFAULT '' COMMENT '支付结果的回调地址',
    `refund_notify_url`   varchar(1024) NULL     DEFAULT '' COMMENT '退款结果的回调地址',
    `transfer_notify_url` varchar(1024) NULL     DEFAULT '' COMMENT '转账结果的回调地址',
    `creator`             varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_app_key` (`app_key` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付应用表';

-- ----------------------------
-- Table structure for pay_channel
-- ----------------------------
DROP TABLE IF EXISTS `pay_channel`;
CREATE TABLE `pay_channel`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '渠道编号',
    `code`        varchar(64)    NOT NULL COMMENT '渠道编码: 1-微信支付, 2-支付宝支付, 3-银联支付',
    `status`      tinyint        NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `fee_rate`    decimal(10, 2) NOT NULL COMMENT '渠道费率，单位：百分比',
    `remark`      varchar(512)   NOT NULL COMMENT '备注',
    `app_id`      bigint         NOT NULL COMMENT '应用编号',
    `config`      text           NULL COMMENT '支付渠道配置',
    `creator`     varchar(64)    NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)    NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`   bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付渠道表';
-- ----------------------------
-- Table structure for pay_demo_order
-- ----------------------------
DROP TABLE IF EXISTS `pay_demo_order`;
CREATE TABLE `pay_demo_order`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '订单编号',
    `user_id`          bigint       NOT NULL COMMENT '用户编号',
    `spu_id`           bigint       NOT NULL COMMENT '商品编号',
    `spu_name`         varchar(255) NOT NULL COMMENT '商品名称',
    `price`            int          NOT NULL COMMENT '价格，单位：分',
    `pay_status`       bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否支付: 0-未支付, 1-已支付',
    `pay_order_id`     bigint       NOT NULL COMMENT '支付订单编号',
    `pay_time`         datetime     NOT NULL COMMENT '付款时间',
    `pay_channel_code` varchar(64)  NOT NULL COMMENT '支付渠道编码',
    `pay_refund_id`    bigint       NOT NULL COMMENT '支付退款单号',
    `refund_price`     int          NOT NULL COMMENT '退款金额，单位：分',
    `refund_time`      datetime     NOT NULL COMMENT '退款完成时间',
    `creator`          varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '示例订单表';
-- ----------------------------
-- Table structure for pay_demo_withdraw
-- ----------------------------
DROP TABLE IF EXISTS `pay_demo_withdraw`;
CREATE TABLE `pay_demo_withdraw`
(
    `id`                    bigint       NOT NULL AUTO_INCREMENT COMMENT '提现单编号',
    `subject`               varchar(255) NOT NULL COMMENT '提现标题',
    `price`                 int          NOT NULL COMMENT '提现金额，单位：分',
    `user_account`          varchar(64)  NOT NULL COMMENT '收款人账号',
    `user_name`             varchar(64)  NOT NULL COMMENT '收款人姓名',
    `type`                  tinyint      NOT NULL COMMENT '提现方式: 1-银行卡, 2-微信, 3-支付宝',
    `status`                tinyint      NOT NULL COMMENT '提现状态: 0-待处理, 1-成功, 2-失败',
    `pay_transfer_id`       bigint       NOT NULL COMMENT '转账单编号',
    `transfer_channel_code` varchar(64)  NOT NULL COMMENT '转账渠道编码',
    `transfer_time`         datetime     NOT NULL COMMENT '转账成功时间',
    `transfer_error_msg`    varchar(255) NOT NULL COMMENT '转账错误提示',
    `creator`               varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`           datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`               varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`           datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`               bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '示例提现订单表';
-- ----------------------------
-- Table structure for pay_notify_log
-- ----------------------------
DROP TABLE IF EXISTS `pay_notify_log`;
CREATE TABLE `pay_notify_log`
(
    `id`           bigint        NOT NULL AUTO_INCREMENT COMMENT '日志编号',
    `task_id`      bigint        NOT NULL COMMENT '通知任务编号',
    `notify_times` int           NOT NULL COMMENT '第几次被通知',
    `response`     varchar(1024) NOT NULL COMMENT 'HTTP响应结果',
    `status`       tinyint       NOT NULL COMMENT '支付通知状态: 0-成功, 1-失败',
    `creator`      varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付通知日志表';
-- ----------------------------
-- Table structure for pay_notify_task
-- ----------------------------
DROP TABLE IF EXISTS `pay_notify_task`;
CREATE TABLE `pay_notify_task`
(
    `id`                   bigint        NOT NULL AUTO_INCREMENT COMMENT '任务编号',
    `app_id`               bigint        NOT NULL COMMENT '应用编号',
    `type`                 tinyint       NOT NULL COMMENT '通知类型: 1-支付, 2-退款, 3-转账',
    `data_id`              bigint        NOT NULL COMMENT '数据编号',
    `merchant_order_id`    varchar(64)   NOT NULL COMMENT '商户订单编号',
    `merchant_refund_id`   varchar(64)   NOT NULL COMMENT '商户退款编号',
    `merchant_transfer_id` varchar(64)   NOT NULL COMMENT '商户转账编号',
    `status`               tinyint       NOT NULL COMMENT '通知状态: 0-等待, 1-成功, 2-失败',
    `next_notify_time`     datetime      NOT NULL COMMENT '下次通知时间',
    `last_execute_time`    datetime      NOT NULL COMMENT '最后一次执行时间',
    `notify_times`         int           NOT NULL COMMENT '当前通知次数',
    `max_notify_times`     int           NOT NULL COMMENT '最大可通知次数',
    `notify_url`           varchar(1024) NOT NULL COMMENT '通知地址',
    `creator`              varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`            bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付通知任务表';

-- ----------------------------
-- Table structure for pay_order
-- ----------------------------
DROP TABLE IF EXISTS `pay_order`;
CREATE TABLE `pay_order`
(
    `id`                bigint         NOT NULL AUTO_INCREMENT COMMENT '订单编号',
    `app_id`            bigint         NOT NULL COMMENT '应用编号',
    `channel_id`        bigint         NOT NULL COMMENT '渠道编号',
    `channel_code`      varchar(64)    NOT NULL COMMENT '渠道编码: 1-微信支付, 2-支付宝支付, 3-银联支付',
    `merchant_order_id` varchar(64)    NOT NULL COMMENT '商户订单编号',
    `subject`           varchar(255)   NOT NULL COMMENT '商品标题',
    `body`              varchar(512)   NOT NULL COMMENT '商品描述信息',
    `notify_url`        varchar(1024)  NOT NULL COMMENT '异步通知地址',
    `price`             int            NOT NULL COMMENT '支付金额，单位：分',
    `channel_fee_rate`  decimal(10, 2) NOT NULL COMMENT '渠道手续费，单位：百分比',
    `channel_fee_price` int            NOT NULL COMMENT '渠道手续金额，单位：分',
    `status`            tinyint        NOT NULL COMMENT '支付状态: 0-待支付, 1-已支付, 2-已取消',
    `user_ip`           varchar(50)    NOT NULL COMMENT '用户IP',
    `expire_time`       datetime       NOT NULL COMMENT '订单失效时间',
    `success_time`      datetime       NOT NULL COMMENT '订单支付成功时间',
    `extension_id`      bigint         NOT NULL COMMENT '支付成功的订单拓展单编号',
    `no`                varchar(64)    NOT NULL COMMENT '外部订单号',
    `refund_price`      int            NOT NULL COMMENT '退款总金额，单位：分',
    `channel_user_id`   varchar(64)    NOT NULL COMMENT '渠道用户编号',
    `channel_order_no`  varchar(64)    NOT NULL COMMENT '渠道订单号',
    `creator`           varchar(64)    NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)    NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付订单表';


-- ----------------------------
-- Table structure for pay_order_extension
-- ----------------------------
DROP TABLE IF EXISTS `pay_order_extension`;
CREATE TABLE `pay_order_extension`
(
    `id`                  bigint        NOT NULL AUTO_INCREMENT COMMENT '订单拓展编号',
    `no`                  varchar(64)   NOT NULL COMMENT '外部订单号',
    `order_id`            bigint        NOT NULL COMMENT '订单号',
    `channel_id`          bigint        NOT NULL COMMENT '渠道编号',
    `channel_code`        varchar(64)   NOT NULL COMMENT '渠道编码: 1-微信支付, 2-支付宝支付, 3-银联支付',
    `user_ip`             varchar(50)   NOT NULL COMMENT '用户IP',
    `status`              tinyint       NOT NULL COMMENT '支付状态: 0-待支付, 1-已支付, 2-已取消',
    `channel_extras`      text          NULL COMMENT '支付渠道的额外参数',
    `channel_error_code`  varchar(64)   NOT NULL COMMENT '调用渠道的错误码',
    `channel_error_msg`   varchar(255)  NOT NULL COMMENT '调用渠道报错时，错误信息',
    `channel_notify_data` varchar(1024) NOT NULL COMMENT '支付渠道的同步/异步通知的内容',
    `creator`             varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付订单拓展表';
-- ----------------------------
-- Table structure for pay_refund
-- ----------------------------
DROP TABLE IF EXISTS `pay_refund`;
CREATE TABLE `pay_refund`
(
    `id`                  bigint        NOT NULL AUTO_INCREMENT COMMENT '退款单编号',
    `no`                  varchar(64)   NOT NULL COMMENT '外部退款号',
    `app_id`              bigint        NOT NULL COMMENT '应用编号',
    `channel_id`          bigint        NOT NULL COMMENT '渠道编号',
    `channel_code`        varchar(64)   NOT NULL COMMENT '渠道编码: 1-微信支付, 2-支付宝支付, 3-银联支付',
    `order_id`            bigint        NOT NULL COMMENT '订单编号',
    `order_no`            varchar(64)   NOT NULL COMMENT '支付订单编号',
    `merchant_order_id`   varchar(64)   NOT NULL COMMENT '商户订单编号',
    `merchant_refund_id`  varchar(64)   NOT NULL COMMENT '商户退款订单号',
    `notify_url`          varchar(1024) NOT NULL COMMENT '异步通知地址',
    `status`              tinyint       NOT NULL COMMENT '退款状态: 0-处理中, 1-成功, 2-失败',
    `pay_price`           int           NOT NULL COMMENT '支付金额，单位：分',
    `refund_price`        int           NOT NULL COMMENT '退款金额，单位：分',
    `reason`              varchar(512)  NOT NULL COMMENT '退款原因',
    `user_ip`             varchar(50)   NOT NULL COMMENT '用户IP',
    `channel_order_no`    varchar(64)   NOT NULL COMMENT '渠道订单号',
    `channel_refund_no`   varchar(64)   NOT NULL COMMENT '渠道退款单号',
    `success_time`        datetime      NOT NULL COMMENT '退款成功时间',
    `channel_error_code`  varchar(64)   NOT NULL COMMENT '调用渠道的错误码',
    `channel_error_msg`   varchar(255)  NOT NULL COMMENT '调用渠道的错误提示',
    `channel_notify_data` varchar(1024) NOT NULL COMMENT '支付渠道的同步/异步通知的内容',
    `creator`             varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '支付退款单表';

-- ----------------------------
-- Table structure for pay_transfer
-- ----------------------------
DROP TABLE IF EXISTS `pay_transfer`;
CREATE TABLE `pay_transfer`
(
    `id`                   bigint        NOT NULL AUTO_INCREMENT COMMENT '转账单编号',
    `no`                   varchar(64)   NOT NULL COMMENT '转账单号',
    `app_id`               bigint        NOT NULL COMMENT '应用编号',
    `channel_id`           bigint        NOT NULL COMMENT '转账渠道编号',
    `channel_code`         varchar(64)   NOT NULL COMMENT '转账渠道编码: 1-微信支付, 2-支付宝支付, 3-银联支付',
    `merchant_transfer_id` varchar(64)   NOT NULL COMMENT '商户转账单编号',
    `subject`              varchar(255)  NOT NULL COMMENT '转账标题',
    `price`                int           NOT NULL COMMENT '转账金额，单位：分',
    `user_account`         varchar(64)   NOT NULL COMMENT '收款人账号',
    `user_name`            varchar(64)   NOT NULL COMMENT '收款人姓名',
    `status`               tinyint       NOT NULL COMMENT '转账状态: 0-处理中, 1-成功, 2-失败',
    `success_time`         datetime      NOT NULL COMMENT '订单转账成功时间',
    `notify_url`           varchar(1024) NOT NULL COMMENT '异步通知地址',
    `user_ip`              varchar(50)   NOT NULL COMMENT '用户IP',
    `channel_extras`       text          NULL COMMENT '渠道的额外参数',
    `channel_transfer_no`  varchar(64)   NOT NULL COMMENT '渠道转账单号',
    `channel_error_code`   varchar(64)   NOT NULL COMMENT '调用渠道的错误码',
    `channel_error_msg`    varchar(255)  NOT NULL COMMENT '调用渠道的错误提示',
    `channel_notify_data`  varchar(1024) NOT NULL COMMENT '渠道的同步/异步通知的内容',
    `channel_package_info` varchar(255)  NOT NULL COMMENT '渠道package信息',
    `creator`              varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '转账单表';

-- ----------------------------
-- Table structure for pay_wallet
-- ----------------------------
DROP TABLE IF EXISTS `pay_wallet`;
CREATE TABLE `pay_wallet`
(
    `id`             bigint      NOT NULL AUTO_INCREMENT COMMENT '钱包编号',
    `user_id`        bigint      NOT NULL COMMENT '用户编号',
    `user_type`      tinyint     NOT NULL COMMENT '用户类型: 1-会员, 2-管理员',
    `balance`        int         NOT NULL COMMENT '余额，单位分',
    `freeze_price`   int         NOT NULL COMMENT '冻结金额，单位分',
    `total_expense`  int         NOT NULL COMMENT '累计支出，单位分',
    `total_recharge` int         NOT NULL COMMENT '累计充值，单位分',
    `creator`        varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员钱包表';
-- ----------------------------
-- Table structure for pay_wallet_recharge
-- ----------------------------
DROP TABLE IF EXISTS `pay_wallet_recharge`;
CREATE TABLE `pay_wallet_recharge`
(
    `id`                 bigint      NOT NULL AUTO_INCREMENT COMMENT '充值编号',
    `wallet_id`          bigint      NOT NULL COMMENT '钱包编号',
    `total_price`        int         NOT NULL COMMENT '用户实际到账余额',
    `pay_price`          int         NOT NULL COMMENT '实际支付金额',
    `bonus_price`        int         NOT NULL COMMENT '钱包赠送金额',
    `package_id`         bigint      NOT NULL COMMENT '充值套餐编号',
    `pay_status`         bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否已支付: 0-未支付, 1-已支付',
    `pay_order_id`       bigint      NOT NULL COMMENT '支付订单编号',
    `pay_channel_code`   varchar(64) NOT NULL COMMENT '支付成功的支付渠道',
    `pay_time`           datetime    NOT NULL COMMENT '订单支付时间',
    `pay_refund_id`      bigint      NOT NULL COMMENT '支付退款单编号',
    `refund_total_price` int         NOT NULL COMMENT '退款金额（包含赠送）',
    `refund_pay_price`   int         NOT NULL COMMENT '退款支付金额',
    `refund_bonus_price` int         NOT NULL COMMENT '退款钱包赠送金额',
    `refund_time`        datetime    NOT NULL COMMENT '退款时间',
    `refund_status`      tinyint     NOT NULL COMMENT '退款状态: 0-处理中, 1-成功, 2-失败',
    `creator`            varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`        datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`        datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员钱包充值表';

-- ----------------------------
-- Table structure for pay_wallet_recharge_package
-- ----------------------------
DROP TABLE IF EXISTS `pay_wallet_recharge_package`;
CREATE TABLE `pay_wallet_recharge_package`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '套餐编号',
    `name`        varchar(255) NOT NULL COMMENT '套餐名',
    `pay_price`   int          NOT NULL COMMENT '支付金额',
    `bonus_price` int          NOT NULL COMMENT '赠送金额',
    `status`      tinyint      NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员钱包充值套餐表';

-- ----------------------------
-- Table structure for pay_wallet_transaction
-- ----------------------------
DROP TABLE IF EXISTS `pay_wallet_transaction`;
CREATE TABLE `pay_wallet_transaction`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '流水编号',
    `no`          varchar(64)  NOT NULL COMMENT '流水号',
    `wallet_id`   bigint       NOT NULL COMMENT '钱包编号',
    `biz_type`    tinyint      NOT NULL COMMENT '关联业务分类: 1-充值, 2-消费, 3-转账, 4-提现',
    `biz_id`      varchar(64)  NOT NULL COMMENT '关联业务编号',
    `title`       varchar(255) NOT NULL COMMENT '流水说明',
    `price`       int          NOT NULL COMMENT '交易金额，单位分（正值表示增加，负值表示减少）',
    `balance`     int          NOT NULL COMMENT '交易后余额，单位分',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '会员钱包流水表';
