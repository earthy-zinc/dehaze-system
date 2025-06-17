CREATE DATABASE IF NOT EXISTS `pei_mall_trade`;
USE `pei_mall_trade`;

-- ----------------------------
-- Table structure for trade_after_sale
-- ----------------------------
DROP TABLE IF EXISTS `trade_after_sale`;
CREATE TABLE `trade_after_sale`
(
    `id`                bigint        NOT NULL AUTO_INCREMENT COMMENT '售后编号',
    `no`                varchar(32)   NOT NULL COMMENT '售后单号',
    `status`            tinyint       NOT NULL COMMENT '退款状态: 0-待处理, 1-已同意, 2-已拒绝, 3-已取消, 4-已完成',
    `way`               tinyint       NOT NULL COMMENT '售后方式: 1-仅退款, 2-退货退款',
    `type`              tinyint       NOT NULL COMMENT '售后类型: 1-质量问题, 2-发错货, 3-其他',
    `user_id`           bigint        NOT NULL COMMENT '用户编号',
    `apply_reason`      varchar(255)  NOT NULL COMMENT '申请原因',
    `apply_description` varchar(512)  NOT NULL COMMENT '补充描述',
    `apply_pic_urls`    text          NULL COMMENT '补充凭证图片列表',
    `order_id`          bigint        NOT NULL COMMENT '交易订单编号',
    `order_no`          varchar(64)   NOT NULL COMMENT '订单流水号',
    `order_item_id`     bigint        NOT NULL COMMENT '交易订单项编号',
    `spu_id`            bigint        NOT NULL COMMENT '商品SPU编号',
    `spu_name`          varchar(255)  NOT NULL COMMENT '商品SPU名称',
    `sku_id`            bigint        NOT NULL COMMENT '商品SKU编号',
    `properties`        text          NULL COMMENT '属性数组',
    `pic_url`           varchar(1024) NOT NULL COMMENT '商品图片',
    `count`             int           NOT NULL COMMENT '退货商品数量',
    `audit_time`        datetime      NOT NULL COMMENT '审批时间',
    `audit_user_id`     bigint        NOT NULL COMMENT '审批人',
    `audit_reason`      varchar(255)  NOT NULL COMMENT '审批备注',
    `refund_price`      int           NOT NULL COMMENT '退款金额，单位：分',
    `pay_refund_id`     bigint        NOT NULL COMMENT '支付退款编号',
    `refund_time`       datetime      NOT NULL COMMENT '退款时间',
    `logistics_id`      bigint        NOT NULL COMMENT '退货物流公司编号',
    `logistics_no`      varchar(64)   NOT NULL COMMENT '退货物流单号',
    `delivery_time`     datetime      NOT NULL COMMENT '退货时间',
    `receive_time`      datetime      NOT NULL COMMENT '收货时间',
    `receive_reason`    varchar(255)  NOT NULL COMMENT '收货备注',
    `creator`           varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '售后订单表';

-- ----------------------------
-- Table structure for trade_after_sale_log
-- ----------------------------
DROP TABLE IF EXISTS `trade_after_sale_log`;
CREATE TABLE `trade_after_sale_log`
(
    `id`            bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`       bigint       NOT NULL COMMENT '用户编号',
    `user_type`     tinyint      NOT NULL COMMENT '用户类型: 1-后台管理, 2-前台用户',
    `after_sale_id` bigint       NOT NULL COMMENT '售后编号',
    `before_status` tinyint      NOT NULL COMMENT '操作前状态',
    `after_status`  tinyint      NOT NULL COMMENT '操作后状态',
    `operate_type`  tinyint      NOT NULL COMMENT '操作类型: 1-提交申请, 2-审核通过, 3-审核拒绝, 4-确认退款, 5-确认收货',
    `content`       varchar(512) NOT NULL COMMENT '操作明细',
    `creator`       varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (id) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '售后日志表';


-- ----------------------------
-- Table structure for trade_brokerage_record
-- ----------------------------
DROP TABLE IF EXISTS `trade_brokerage_record`;
CREATE TABLE `trade_brokerage_record`
(
    `id`                int          NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`           bigint       NOT NULL COMMENT '用户编号',
    `biz_id`            varchar(64)  NOT NULL COMMENT '业务编号',
    `biz_type`          tinyint      NOT NULL COMMENT '业务类型: 1-订单佣金, 2-提现手续费返还',
    `title`             varchar(255) NOT NULL COMMENT '标题',
    `description`       varchar(512) NOT NULL COMMENT '说明',
    `price`             int          NOT NULL COMMENT '金额，单位：分',
    `total_price`       int          NOT NULL COMMENT '当前总佣金，单位：分',
    `status`            tinyint      NOT NULL COMMENT '状态: 0-未生效, 1-已生效, 2-已失效, 3-已解冻',
    `frozen_days`       int          NOT NULL COMMENT '冻结时间（天）',
    `unfreeze_time`     datetime     NOT NULL COMMENT '解冻时间',
    `source_user_level` int          NOT NULL COMMENT '来源用户等级',
    `source_user_id`    bigint       NOT NULL COMMENT '来源用户编号',
    `creator`           varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '佣金记录表';

-- ----------------------------
-- Table structure for trade_brokerage_user
-- ----------------------------
DROP TABLE IF EXISTS `trade_brokerage_user`;
CREATE TABLE `trade_brokerage_user`
(
    `id`                bigint      NOT NULL COMMENT '用户编号',
    `bind_user_id`      bigint      NOT NULL COMMENT '推广员编号',
    `bind_user_time`    datetime    NOT NULL COMMENT '推广员绑定时间',
    `brokerage_enabled` bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否有分销资格: 0-否, 1-是',
    `brokerage_time`    datetime    NOT NULL COMMENT '成为分销员时间',
    `brokerage_price`   int         NOT NULL COMMENT '可用佣金，单位：分',
    `frozen_price`      int         NOT NULL COMMENT '冻结佣金，单位：分',
    `creator`           varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '分销用户表';

-- ----------------------------
-- Table structure for trade_brokerage_withdraw
-- ----------------------------
DROP TABLE IF EXISTS `trade_brokerage_withdraw`;
CREATE TABLE `trade_brokerage_withdraw`
(
    `id`                    bigint        NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`               bigint        NOT NULL COMMENT '用户编号',
    `price`                 int           NOT NULL COMMENT '提现金额，单位：分',
    `fee_price`             int           NOT NULL COMMENT '提现手续费，单位：分',
    `total_price`           int           NOT NULL COMMENT '当前总佣金，单位：分',
    `type`                  tinyint       NOT NULL COMMENT '提现类型: 1-银行卡, 2-微信, 3-支付宝',
    `user_name`             varchar(64)   NOT NULL COMMENT '提现姓名',
    `user_account`          varchar(64)   NOT NULL COMMENT '提现账号',
    `qr_code_url`           varchar(1024) NOT NULL COMMENT '收款码地址',
    `bank_name`             varchar(64)   NOT NULL COMMENT '银行名称',
    `bank_address`          varchar(128)  NOT NULL COMMENT '开户地址',
    `status`                tinyint       NOT NULL COMMENT '状态: 0-待审核, 1-已通过, 2-已驳回, 3-已转账',
    `audit_reason`          varchar(255)  NOT NULL COMMENT '审核驳回原因',
    `audit_time`            datetime      NOT NULL COMMENT '审核时间',
    `remark`                varchar(512)  NOT NULL COMMENT '备注',
    `pay_transfer_id`       bigint        NOT NULL COMMENT '转账单编号',
    `transfer_channel_code` varchar(64)   NOT NULL COMMENT '转账渠道编码',
    `transfer_time`         datetime      NOT NULL COMMENT '转账成功时间',
    `transfer_error_msg`    varchar(255)  NOT NULL COMMENT '转账错误提示',
    `creator`               varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`           datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`               varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`           datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`               bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '佣金提现表';

-- ----------------------------
-- Table structure for trade_cart
-- ----------------------------
DROP TABLE IF EXISTS `trade_cart`;
CREATE TABLE `trade_cart`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '购物车编号',
    `user_id`     bigint      NOT NULL COMMENT '用户编号',
    `spu_id`      bigint      NOT NULL COMMENT '商品SPU编号',
    `sku_id`      bigint      NOT NULL COMMENT '商品SKU编号',
    `count`       int         NOT NULL COMMENT '商品购买数量',
    `selected`    bit(1)      NOT NULL DEFAULT b'1' COMMENT '是否选中: 0-未选中, 1-选中',
    `creator`     varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '购物车商品表';

-- ----------------------------
-- Table structure for trade_config
-- ----------------------------
DROP TABLE IF EXISTS `trade_config`;
CREATE TABLE `trade_config`
(
    `id`                             bigint      NOT NULL AUTO_INCREMENT COMMENT '配置编号',
    `after_sale_refund_reasons`      text        NULL COMMENT '售后退款理由列表',
    `after_sale_return_reasons`      text        NULL COMMENT '售后退货理由列表',
    `delivery_express_free_enabled`  bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否启用全场包邮: 0-否, 1-是',
    `delivery_express_free_price`    int         NOT NULL COMMENT '全场包邮的最小金额，单位：分',
    `delivery_pick_up_enabled`       bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否开启自提: 0-否, 1-是',
    `brokerage_enabled`              bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否启用分佣: 0-否, 1-是',
    `brokerage_enabled_condition`    tinyint     NOT NULL COMMENT '分佣模式: 1-所有订单, 2-指定商品',
    `brokerage_bind_mode`            tinyint     NOT NULL COMMENT '分销关系绑定模式: 1-首单绑定, 2-点击绑定, 3-两者优先',
    `brokerage_poster_urls`          text        NULL COMMENT '分销海报图地址列表',
    `brokerage_first_percent`        int         NOT NULL COMMENT '一级返佣比例',
    `brokerage_second_percent`       int         NOT NULL COMMENT '二级返佣比例',
    `brokerage_withdraw_min_price`   int         NOT NULL COMMENT '用户提现最低金额',
    `brokerage_withdraw_fee_percent` int         NOT NULL COMMENT '用户提现手续费百分比',
    `brokerage_frozen_days`          int         NOT NULL COMMENT '佣金冻结时间（天）',
    `brokerage_withdraw_types`       text        NOT NULL COMMENT '提现方式列表: 1-银行卡, 2-微信, 3-支付宝',
    `creator`                        varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`                    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                        varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`                    datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                        bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '交易中心配置表';

-- ----------------------------
-- Table structure for trade_delivery_express
-- ----------------------------
DROP TABLE IF EXISTS `trade_delivery_express`;
CREATE TABLE `trade_delivery_express`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '快递公司编号',
    `code`        varchar(64)   NOT NULL COMMENT '快递公司code',
    `name`        varchar(255)  NOT NULL COMMENT '快递公司名称',
    `logo`        varchar(1024) NOT NULL COMMENT '快递公司logo',
    `sort`        int           NOT NULL COMMENT '排序',
    `status`      tinyint       NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `creator`     varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '快递公司表';

-- ----------------------------
-- Table structure for trade_delivery_express_template_charge
-- ----------------------------
DROP TABLE IF EXISTS `trade_delivery_express_template_charge`;
CREATE TABLE `trade_delivery_express_template_charge`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '编号',
    `template_id` bigint         NOT NULL COMMENT '配送模板编号',
    `area_ids`    text           NOT NULL COMMENT '配送区域编号列表',
    `charge_mode` tinyint        NOT NULL COMMENT '配送计费方式: 1-按件数, 2-按重量, 3-按体积',
    `start_count` decimal(10, 2) NOT NULL COMMENT '首件数量',
    `start_price` int            NOT NULL COMMENT '起步价，单位：分',
    `extra_count` decimal(10, 2) NOT NULL COMMENT '续件数量',
    `extra_price` int            NOT NULL COMMENT '额外价，单位：分',
    `creator`     varchar(64)    NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)    NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '快递运费模板计费配置表';

-- ----------------------------
-- Table structure for trade_delivery_express_template
-- ----------------------------
DROP TABLE IF EXISTS `trade_delivery_express_template`;
CREATE TABLE `trade_delivery_express_template`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '模板编号',
    `name`        varchar(255) NOT NULL COMMENT '模板名称',
    `charge_mode` tinyint      NOT NULL COMMENT '配送计费方式: 1-按件数, 2-按重量, 3-按体积',
    `sort`        int          NOT NULL COMMENT '排序',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '快递运费模板表';

-- ----------------------------
-- Table structure for trade_delivery_express_template_free
-- ----------------------------
DROP TABLE IF EXISTS `trade_delivery_express_template_free`;
CREATE TABLE `trade_delivery_express_template_free`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `template_id` bigint      NOT NULL COMMENT '配送模板编号',
    `area_ids`    text        NOT NULL COMMENT '配送区域编号列表',
    `free_price`  int         NOT NULL COMMENT '包邮金额，单位：分',
    `free_count`  int         NOT NULL COMMENT '包邮件数',
    `creator`     varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '快递运费模板包邮配置表';


-- ----------------------------
-- Table structure for trade_delivery_pick_up_store
-- ----------------------------
DROP TABLE IF EXISTS `trade_delivery_pick_up_store`;
CREATE TABLE `trade_delivery_pick_up_store`
(
    `id`              bigint        NOT NULL AUTO_INCREMENT COMMENT '编号',
    `name`            varchar(255)  NOT NULL COMMENT '门店名称',
    `introduction`    varchar(512)  NOT NULL COMMENT '门店简介',
    `phone`           varchar(32)   NOT NULL COMMENT '门店手机',
    `area_id`         int           NOT NULL COMMENT '区域编号',
    `detail_address`  varchar(255)  NOT NULL COMMENT '门店详细地址',
    `logo`            varchar(1024) NOT NULL COMMENT '门店logo',
    `opening_time`    time          NOT NULL COMMENT '营业开始时间',
    `closing_time`    time          NOT NULL COMMENT '营业结束时间',
    `latitude`        decimal(9, 6) NOT NULL COMMENT '纬度',
    `longitude`       decimal(9, 6) NOT NULL COMMENT '经度',
    `verify_user_ids` text          NOT NULL COMMENT '核销员工用户编号列表',
    `status`          tinyint       NOT NULL COMMENT '门店状态: 0-开启, 1-禁用',
    `creator`         varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`     datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`     datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '自提门店表';

-- ----------------------------
-- Table structure for trade_order
-- ----------------------------
DROP TABLE IF EXISTS `trade_order`;
CREATE TABLE `trade_order`
(
    `id`                          bigint       NOT NULL AUTO_INCREMENT COMMENT '订单编号',
    `no`                          varchar(64)  NOT NULL COMMENT '订单流水号',
    `type`                        tinyint      NOT NULL COMMENT '订单类型: 1-普通订单, 2-秒杀订单, 3-砍价订单, 4-拼团订单, 5-积分商城订单',
    `terminal`                    tinyint      NOT NULL COMMENT '订单来源: 1-PC, 2-移动端, 3-小程序',
    `user_id`                     bigint       NOT NULL COMMENT '用户编号',
    `user_ip`                     varchar(50)  NOT NULL COMMENT '用户IP',
    `user_remark`                 varchar(512) NOT NULL COMMENT '用户备注',
    `status`                      tinyint      NOT NULL COMMENT '订单状态: 0-待支付, 1-已支付, 2-已发货, 3-已完成, 4-已关闭',
    `product_count`               int          NOT NULL COMMENT '购买的商品数量',
    `finish_time`                 datetime     NOT NULL COMMENT '订单完成时间',
    `cancel_time`                 datetime     NOT NULL COMMENT '订单取消时间',
    `cancel_type`                 tinyint      NOT NULL COMMENT '取消类型: 1-用户取消, 2-超时未支付自动取消, 3-系统取消',
    `remark`                      varchar(512) NOT NULL COMMENT '商家备注',
    `comment_status`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否评价: 0-未评价, 1-已评价',
    `brokerage_user_id`           bigint       NOT NULL COMMENT '推广人编号',
    `pay_order_id`                bigint       NOT NULL COMMENT '支付订单编号',
    `pay_status`                  bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否已支付: 0-否, 1-是',
    `pay_time`                    datetime     NOT NULL COMMENT '付款时间',
    `pay_channel_code`            varchar(64)  NOT NULL COMMENT '支付渠道编码',
    `total_price`                 int          NOT NULL COMMENT '商品原价，单位：分',
    `discount_price`              int          NOT NULL COMMENT '优惠金额，单位：分',
    `delivery_price`              int          NOT NULL COMMENT '运费金额，单位：分',
    `adjust_price`                int          NOT NULL COMMENT '订单调价，单位：分',
    `pay_price`                   int          NOT NULL COMMENT '应付金额（总），单位：分',
    `delivery_type`               tinyint      NOT NULL COMMENT '配送方式: 1-快递, 2-自提',
    `logistics_id`                bigint       NOT NULL COMMENT '发货物流公司编号: 0-无需发货',
    `logistics_no`                varchar(64)  NOT NULL COMMENT '发货物流单号',
    `delivery_time`               datetime     NOT NULL COMMENT '发货时间',
    `receive_time`                datetime     NOT NULL COMMENT '收货时间',
    `receiver_name`               varchar(64)  NOT NULL COMMENT '收件人名称',
    `receiver_mobile`             varchar(32)  NOT NULL COMMENT '收件人手机',
    `receiver_area_id`            int          NOT NULL COMMENT '收件人地区编号',
    `receiver_detail_address`     varchar(255) NOT NULL COMMENT '收件人详细地址',
    `pick_up_store_id`            bigint       NOT NULL COMMENT '自提门店编号',
    `pick_up_verify_code`         varchar(32)  NOT NULL COMMENT '自提核销码',
    `refund_status`               tinyint      NOT NULL COMMENT '售后状态: 1-无售后, 2-有售后',
    `refund_price`                int          NOT NULL COMMENT '退款金额，单位：分',
    `coupon_id`                   bigint       NOT NULL COMMENT '优惠劵编号',
    `coupon_price`                int          NOT NULL COMMENT '优惠劵减免金额，单位：分',
    `use_point`                   int          NOT NULL COMMENT '使用的积分',
    `point_price`                 int          NOT NULL COMMENT '积分抵扣的金额，单位：分',
    `give_point`                  int          NOT NULL COMMENT '赠送的积分',
    `refund_point`                int          NOT NULL COMMENT '退还的使用的积分',
    `vip_price`                   int          NOT NULL COMMENT 'VIP减免金额，单位：分',
    `give_coupon_template_counts` text         NULL COMMENT '赠送的优惠劵模板计数',
    `give_coupon_ids`             text         NULL COMMENT '赠送的优惠劵编号列表',
    `seckill_activity_id`         bigint       NOT NULL COMMENT '秒杀活动编号',
    `bargain_activity_id`         bigint       NOT NULL COMMENT '砍价活动编号',
    `bargain_record_id`           bigint       NOT NULL COMMENT '砍价记录编号',
    `combination_activity_id`     bigint       NOT NULL COMMENT '拼团活动编号',
    `combination_head_id`         bigint       NOT NULL COMMENT '拼团团长编号',
    `combination_record_id`       bigint       NOT NULL COMMENT '拼团记录编号',
    `point_activity_id`           bigint       NOT NULL COMMENT '积分商城活动编号',
    `creator`                     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`                 datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`                 datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '交易订单表';

-- ----------------------------
-- Table structure for trade_order_item
-- ----------------------------
DROP TABLE IF EXISTS `trade_order_item`;
CREATE TABLE `trade_order_item`
(
    `id`                bigint        NOT NULL AUTO_INCREMENT COMMENT '订单项编号',
    `user_id`           bigint        NOT NULL COMMENT '用户编号',
    `order_id`          bigint        NOT NULL COMMENT '订单编号',
    `cart_id`           bigint        NOT NULL COMMENT '购物车项编号',
    `spu_id`            bigint        NOT NULL COMMENT '商品SPU编号',
    `spu_name`          varchar(255)  NOT NULL COMMENT '商品SPU名称',
    `sku_id`            bigint        NOT NULL COMMENT '商品SKU编号',
    `properties`        text          NULL COMMENT '属性数组',
    `pic_url`           varchar(1024) NOT NULL COMMENT '商品图片',
    `count`             int           NOT NULL COMMENT '购买数量',
    `comment_status`    bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否评价: 0-否, 1-是',
    `price`             int           NOT NULL COMMENT '商品原价（单），单位：分',
    `discount_price`    int           NOT NULL COMMENT '优惠金额（总），单位：分',
    `delivery_price`    int           NOT NULL COMMENT '运费金额（总），单位：分',
    `adjust_price`      int           NOT NULL COMMENT '订单调价（总），单位：分',
    `pay_price`         int           NOT NULL COMMENT '应付金额（总），单位：分',
    `coupon_price`      int           NOT NULL COMMENT '优惠劵减免金额，单位：分',
    `point_price`       int           NOT NULL COMMENT '积分抵扣的金额，单位：分',
    `use_point`         int           NOT NULL COMMENT '使用的积分',
    `give_point`        int           NOT NULL COMMENT '赠送的积分',
    `vip_price`         int           NOT NULL COMMENT 'VIP减免金额，单位：分',
    `after_sale_id`     bigint        NOT NULL COMMENT '售后单编号',
    `after_sale_status` tinyint       NOT NULL COMMENT '售后状态: 0-无售后, 1-售后中, 2-售后完成',
    `creator`           varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '交易订单项表';

-- ----------------------------
-- Table structure for trade_order_log
-- ----------------------------
DROP TABLE IF EXISTS `trade_order_log`;
CREATE TABLE `trade_order_log`
(
    `id`            bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`       bigint       NOT NULL COMMENT '用户编号',
    `user_type`     tinyint      NOT NULL COMMENT '用户类型: 0-系统, 1-管理员, 2-用户',
    `order_id`      bigint       NOT NULL COMMENT '订单号',
    `before_status` tinyint      NOT NULL COMMENT '操作前状态',
    `after_status`  tinyint      NOT NULL COMMENT '操作后状态',
    `operate_type`  tinyint      NOT NULL COMMENT '操作类型: 1-创建订单, 2-支付订单, 3-发货, 4-完成订单, 5-取消订单',
    `content`       varchar(512) NOT NULL COMMENT '订单日志信息',
    `creator`       varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`   datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '订单日志表';
