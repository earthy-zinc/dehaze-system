CREATE DATABASE IF NOT EXISTS `pei_mall_statistics`;
USE `pei_mall_statistics`;

-- ----------------------------
-- Table structure for product_statistics
-- ----------------------------
DROP TABLE IF EXISTS `product_statistics`;
CREATE TABLE `product_statistics`
(
    `id`                      bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `time`                    date        NOT NULL COMMENT '统计日期',
    `spu_id`                  bigint      NOT NULL COMMENT '商品SPU编号',
    `browse_count`            int         NOT NULL COMMENT '浏览量',
    `browse_user_count`       int         NOT NULL COMMENT '访客量',
    `favorite_count`          int         NOT NULL COMMENT '收藏数量',
    `cart_count`              int         NOT NULL COMMENT '加购数量',
    `order_count`             int         NOT NULL COMMENT '下单件数',
    `order_pay_count`         int         NOT NULL COMMENT '支付件数',
    `order_pay_price`         int         NOT NULL COMMENT '支付金额，单位：分',
    `after_sale_count`        int         NOT NULL COMMENT '退款件数',
    `after_sale_refund_price` int         NOT NULL COMMENT '退款金额，单位：分',
    `browse_convert_percent`  int         NOT NULL COMMENT '访客支付转化率（百分比）',
    `creator`                 varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`             datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                 varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`             datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                 bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '商品统计表';

-- ----------------------------
-- Table structure for trade_statistics
-- ----------------------------
DROP TABLE IF EXISTS `trade_statistics`;
CREATE TABLE `trade_statistics`
(
    `id`                         bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `time`                       datetime    NOT NULL COMMENT '统计日期',
    `order_create_count`         int         NOT NULL COMMENT '创建订单数',
    `order_pay_count`            int         NOT NULL COMMENT '支付订单商品数',
    `order_pay_price`            int         NOT NULL COMMENT '总支付金额，单位：分',
    `after_sale_count`           int         NOT NULL COMMENT '退款订单数',
    `after_sale_refund_price`    int         NOT NULL COMMENT '总退款金额，单位：分',
    `brokerage_settlement_price` int         NOT NULL COMMENT '佣金金额（已结算），单位：分',
    `wallet_pay_price`           int         NOT NULL COMMENT '总支付金额（余额），单位：分',
    `recharge_pay_count`         int         NOT NULL COMMENT '充值订单数',
    `recharge_pay_price`         int         NOT NULL COMMENT '充值金额，单位：分',
    `recharge_refund_count`      int         NOT NULL COMMENT '充值退款订单数',
    `recharge_refund_price`      int         NOT NULL COMMENT '充值退款金额，单位：分',
    `creator`                    varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`                datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                    varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`                datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                    bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '交易统计表';
