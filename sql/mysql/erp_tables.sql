CREATE DATABASE IF NOT EXISTS `pei_erp`;
USE `pei_erp`;

-- ----------------------------
-- Table structure for erp_account
-- ----------------------------
DROP TABLE IF EXISTS `erp_account`;
CREATE TABLE `erp_account`
(
    `id`             bigint       NOT NULL AUTO_INCREMENT COMMENT '账户主键',
    `name`           varchar(255) NOT NULL COMMENT '账户名称',
    `no`             varchar(64)  NOT NULL COMMENT '账户编码',
    `remark`         varchar(512)          DEFAULT '' COMMENT '备注',
    `status`         int          NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `sort`           int          NOT NULL DEFAULT '0' COMMENT '排序',
    `default_status` bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否默认',
    `creator`        varchar(64)           DEFAULT '' COMMENT '创建者',
    `create_time`    datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)           DEFAULT '' COMMENT '更新者',
    `update_time`    datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 银行账户表';

-- ----------------------------
-- Table structure for erp_finance_payment
-- ----------------------------
DROP TABLE IF EXISTS `erp_finance_payment`;
CREATE TABLE `erp_finance_payment`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '付款单主键',
    `no`              varchar(64)    NOT NULL COMMENT '付款单号',
    `status`          int            NOT NULL COMMENT '付款审核状态 枚举 10:未审核,20:已审核',
    `payment_time`    datetime       NOT NULL COMMENT '付款时间',
    `finance_user_id` bigint         NOT NULL COMMENT '财务审核人编号 关联 admin_user.id',
    `supplier_id`     bigint         NOT NULL COMMENT '供应商编号 关联 erp_supplier.id',
    `account_id`      bigint         NOT NULL COMMENT '结算账户编号 关联 erp_account.id',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总金额',
    `discount_price`  decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `payment_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '付款金额',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_supplier_id` (`supplier_id` ASC) USING BTREE,
    INDEX `idx_account_id` (`account_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 财务付款单';

-- ----------------------------
-- Table structure for erp_finance_payment_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_finance_payment_item`;
CREATE TABLE `erp_finance_payment_item`
(
    `id`            bigint         NOT NULL AUTO_INCREMENT COMMENT '付款单项主键',
    `payment_id`    bigint         NOT NULL COMMENT '付款单编号 关联 erp_finance_payment.id',
    `biz_type`      int            NOT NULL COMMENT '业务类型 枚举 70:采购入库,80:采购退货出库',
    `biz_id`        bigint         NOT NULL COMMENT '业务编号',
    `biz_no`        varchar(64)    NOT NULL COMMENT '业务单号',
    `total_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总金额',
    `paid_price`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已付款金额',
    `payment_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '本次付款金额',
    `remark`        varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`       varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`   datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_payment_id` (`payment_id` ASC) USING BTREE,
    INDEX `idx_biz_type` (`biz_type` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 财务付款单项';

-- ----------------------------
-- Table structure for erp_finance_receipt
-- ----------------------------
DROP TABLE IF EXISTS `erp_finance_receipt`;
CREATE TABLE `erp_finance_receipt`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '收款单主键',
    `no`              varchar(64)    NOT NULL COMMENT '收款单号',
    `status`          int            NOT NULL COMMENT '收款审核状态 枚举 10:未审核,20:已审核',
    `receipt_time`    datetime       NOT NULL COMMENT '收款时间',
    `finance_user_id` bigint         NOT NULL COMMENT '财务审核人编号 关联 admin_user.id',

    `account_id`      bigint         NOT NULL COMMENT '收款账户编号 关联 erp_account.id',

    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总金额',
    `discount_price`  decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `receipt_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '实付金额',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_account_id` (`account_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 财务收款单';

-- ----------------------------
-- Table structure for erp_finance_receipt_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_finance_receipt_item`;
CREATE TABLE `erp_finance_receipt_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '收款单项主键',
    `receipt_id`      bigint         NOT NULL COMMENT '收款单编号 关联 erp_finance_receipt.id',
    `biz_type`        int            NOT NULL COMMENT '业务类型 枚举 50:销售出库,60:销售退货入库',
    `biz_id`          bigint         NOT NULL COMMENT '业务编号',
    `biz_no`          varchar(64)    NOT NULL COMMENT '业务单号',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总金额',
    `receipted_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已收款金额',
    `receipt_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '收款金额',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_receipt_id` (`receipt_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 财务收款单项';

-- ----------------------------
-- Table structure for erp_product_category
-- ----------------------------
DROP TABLE IF EXISTS `erp_product_category`;
CREATE TABLE `erp_product_category`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '产品分类主键',
    `name`        varchar(255) NOT NULL COMMENT '分类名称',
    `parent_id`   bigint       NOT NULL DEFAULT '0' COMMENT '父分类编号',
    `code`        varchar(64)  NOT NULL COMMENT '分类编码',
    `sort`        int          NOT NULL DEFAULT '0' COMMENT '排序',
    `status`      int          NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `creator`     varchar(64)           DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)           DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_parent_id` (`parent_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 产品分类表';

-- ----------------------------
-- Table structure for erp_product
-- ----------------------------
DROP TABLE IF EXISTS `erp_product`;
CREATE TABLE `erp_product`
(
    `id`             bigint         NOT NULL AUTO_INCREMENT COMMENT '产品主键',
    `name`           varchar(255)   NOT NULL COMMENT '产品名称',
    `bar_code`       varchar(255)            DEFAULT '' COMMENT '产品条码',
    `category_id`    bigint         NOT NULL COMMENT '产品分类编号 关联 erp_product_category.id',
    `unit_id`        bigint         NOT NULL COMMENT '单位编号 关联 erp_product_unit.id',
    `status`         int            NOT NULL COMMENT '产品状态 枚举 0:开启,1:关闭',
    `standard`       varchar(255)            DEFAULT '' COMMENT '产品规格',
    `remark`         varchar(512)            DEFAULT '' COMMENT '产品备注',
    `expiry_day`     int                     DEFAULT '0' COMMENT '保质期天数',
    `weight`         decimal(10, 2)          DEFAULT '0.00' COMMENT '基础重量（kg）',
    `purchase_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '采购价格，单位：元',
    `sale_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '销售价格，单位：元',
    `min_price`      decimal(10, 2)          DEFAULT '0.00' COMMENT '最低价格，单位：元',
    `creator`        varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_category_id` (`category_id` ASC) USING BTREE,
    INDEX `idx_unit_id` (`unit_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 产品信息';

-- ----------------------------
-- Table structure for erp_product_unit
-- ----------------------------
DROP TABLE IF EXISTS `erp_product_unit`;
CREATE TABLE `erp_product_unit`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '单位主键',
    `name`        varchar(255) NOT NULL COMMENT '单位名称',
    `status`      int          NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `creator`     varchar(64)           DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)           DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 产品单位';

-- ----------------------------
-- Table structure for erp_purchase_in
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_in`;
CREATE TABLE `erp_purchase_in`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '入库单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '入库单号',
    `status`              int            NOT NULL COMMENT '审核状态 枚举 10:未审核,20:已审核',
    `supplier_id`         bigint         NOT NULL COMMENT '供应商编号 关联 erp_supplier.id',
    `account_id`          bigint         NOT NULL COMMENT '结算账户编号 关联 erp_account.id',
    `in_time`             datetime       NOT NULL COMMENT '入库时间',
    `order_id`            bigint         NOT NULL COMMENT '采购订单编号 关联 erp_purchase_order.id',
    `order_no`            varchar(64)    NOT NULL COMMENT '采购订单号',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计金额',
    `payment_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已支付金额',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计产品金额',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计税金金额',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '折扣比例',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '折扣金额',
    `other_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '其他费用金额',
    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_supplier_id` (`supplier_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购入库单';

-- ----------------------------
-- Table structure for erp_purchase_in_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_in_item`;
CREATE TABLE `erp_purchase_in_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '编号',
    `in_id`           bigint         NOT NULL COMMENT '采购入库编号（关联 erp_purchase_in 表的 id 字段）',
    `order_item_id`   bigint         NOT NULL COMMENT '采购订单项编号（关联 erp_purchase_order_item 表的 id 字段）',
    `warehouse_id`    bigint         NOT NULL COMMENT '仓库编号（关联 erp_warehouse 表的 id 字段）',
    `product_id`      bigint         NOT NULL COMMENT '产品编号（关联 erp_product 表的 id 字段）',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号（冗余 erp_product 表的 unit_id 字段）',
    `product_price`   decimal(10, 2) NOT NULL COMMENT '产品单位单价，单位：元',
    `count`           decimal(10, 2) NOT NULL COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL COMMENT '总价，单位：元',
    `tax_percent`     decimal(10, 2) NOT NULL COMMENT '税率，百分比',
    `tax_price`       decimal(10, 2) NOT NULL COMMENT '税额，单位：元',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_in_id` (`in_id` ASC) USING BTREE,
    INDEX `idx_order_item_id` (`order_item_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购入库单项';

-- ----------------------------
-- Table structure for erp_purchase_order
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_order`;
CREATE TABLE `erp_purchase_order`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '采购订单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '订单编号',
    `status`              int            NOT NULL COMMENT '审核状态 枚举 10:未审核,20:已审核',
    `supplier_id`         bigint         NOT NULL COMMENT '供应商编号 关联 erp_supplier.id',
    `account_id`          bigint                  DEFAULT NULL COMMENT '结算账户编号 关联 erp_account.id',
    `order_time`          datetime       NOT NULL COMMENT '下单时间',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '最终合计价格',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计产品价格',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额总金额',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠率',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `deposit_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '定金金额',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `in_count`            decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '入库数量',
    `return_count`        decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '退货数量',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_supplier_id` (`supplier_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购订单';

-- ----------------------------
-- Table structure for erp_purchase_order_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_order_item`;
CREATE TABLE `erp_purchase_order_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '订单项主键',
    `order_id`        bigint         NOT NULL COMMENT '采购订单编号 关联 erp_purchase_order.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总价',
    `tax_percent`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率',
    `tax_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额',
    `in_count`        decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已入库数量',
    `return_count`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已退货数量',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_order_id` (`order_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购订单项';

-- ----------------------------
-- Table structure for erp_purchase_return
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_return`;
CREATE TABLE `erp_purchase_return`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '退货单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '退货单号',
    `status`              int            NOT NULL COMMENT '退货状态 枚举 10:未审核,20:已审核',
    `supplier_id`         bigint         NOT NULL COMMENT '供应商编号 关联 erp_supplier.id',
    `account_id`          bigint         NOT NULL COMMENT '结算账户编号 关联 erp_account.id',
    `return_time`         datetime       NOT NULL COMMENT '退货时间',
    `order_id`            bigint         NOT NULL COMMENT '采购订单编号 关联 erp_purchase_order.id',
    `order_no`            varchar(64)    NOT NULL COMMENT '采购订单号 冗余 erp_purchase_order.no',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '最终合计价格，单位：元 (totalPrice = totalProductPrice + totalTaxPrice - discountPrice + otherPrice)',
    `refund_price`        decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已退款金额，单位：元',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计产品价格，单位：元',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计税额，单位：元',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠率，百分比',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额，单位：元 (discountPrice = (totalProductPrice + totalTaxPrice) * discountPercent)',
    `other_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '其它金额，单位：元',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_supplier_id` (`supplier_id` ASC) USING BTREE,
    INDEX `idx_account_id` (`account_id` ASC) USING BTREE,
    INDEX `idx_order_id` (`order_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购退货单';

-- ----------------------------
-- Table structure for erp_purchase_return_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_purchase_return_item`;
CREATE TABLE `erp_purchase_return_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '退货单项主键',
    `return_id`       bigint         NOT NULL COMMENT '退货单编号 关联 erp_purchase_return.id',
    `order_item_id`   bigint         NOT NULL COMMENT '采购订单项编号 关联 erp_purchase_order_item.id 目的：方便更新关联的采购订单项的退货数量',
    `warehouse_id`    bigint         NOT NULL COMMENT '仓库编号 关联 erp_warehouse.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总价 (totalPrice = productPrice * count)',
    `tax_percent`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率，百分比',
    `tax_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额，单位：元 (taxPrice = totalPrice * taxPercent)',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_return_id` (`return_id` ASC) USING BTREE,
    INDEX `idx_order_item_id` (`order_item_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 采购退货单项';

-- ----------------------------
-- Table structure for erp_supplier
-- ----------------------------
DROP TABLE IF EXISTS `erp_supplier`;
CREATE TABLE `erp_supplier`
(
    `id`           bigint         NOT NULL AUTO_INCREMENT COMMENT '供应商主键',
    `name`         varchar(255)   NOT NULL COMMENT '供应商名称',
    `short_name`   varchar(255)            DEFAULT '' COMMENT '简称',
    `contact`      varchar(64)             DEFAULT '' COMMENT '联系人',
    `mobile`       varchar(32)             DEFAULT '' COMMENT '手机号',
    `telephone`    varchar(32)             DEFAULT '' COMMENT '电话',
    `email`        varchar(64)             DEFAULT '' COMMENT '邮箱',
    `fax`          varchar(32)             DEFAULT '',
    `remark`       varchar(512)            DEFAULT '' COMMENT '备注',
    `status`       int            NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `sort`         int            NOT NULL DEFAULT '0' COMMENT '排序',
    `tax_no`       varchar(64)             DEFAULT '' COMMENT '纳税人识别号',
    `tax_percent`  decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率，百分比',
    `bank_address` varchar(255)            DEFAULT '' COMMENT '开户地址',
    `bank_name`    varchar(255)            DEFAULT '' COMMENT '开户银行',
    `bank_account` varchar(64)             DEFAULT '' COMMENT '银行账号',


    `creator`      varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_status` (`status` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 供应商';

-- ----------------------------
-- Table structure for erp_customer
-- ----------------------------
DROP TABLE IF EXISTS `erp_customer`;
CREATE TABLE `erp_customer`
(
    `id`           bigint         NOT NULL AUTO_INCREMENT COMMENT '客户主键',
    `name`         varchar(255)   NOT NULL COMMENT '客户名称',
    `contact`      varchar(64)             DEFAULT '' COMMENT '联系人',
    `mobile`       varchar(32)             DEFAULT '' COMMENT '手机号',
    `telephone`    varchar(32)             DEFAULT '' COMMENT '电话',
    `email`        varchar(64)             DEFAULT '' COMMENT '邮箱',
    `fax`          varchar(32)             DEFAULT '',
    `remark`       varchar(512)            DEFAULT '' COMMENT '备注',
    `status`       int            NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `sort`         int            NOT NULL DEFAULT '0' COMMENT '排序',
    `tax_no`       varchar(64)             DEFAULT '' COMMENT '税号',
    `tax_percent`  decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率，百分比',
    `bank_address` varchar(255)            DEFAULT '' COMMENT '开户地址',
    `bank_name`    varchar(255)            DEFAULT '' COMMENT '开户银行',
    `bank_account` varchar(64)             DEFAULT '' COMMENT '银行账号',


    `creator`      varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_status` (`status` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 客户';

-- ----------------------------
-- Table structure for erp_sale_order
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_order`;
CREATE TABLE `erp_sale_order`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '销售订单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '订单编号',
    `status`              int            NOT NULL COMMENT '审核状态 枚举 10:未审核,20:已审核',
    `customer_id`         bigint         NOT NULL COMMENT '客户编号 关联 erp_customer.id',
    `account_id`          bigint         NOT NULL COMMENT '账户编号 关联 erp_account.id',
    `sale_user_id`        bigint         NOT NULL COMMENT '销售员编号 关联 admin_user.id',
    `order_time`          datetime       NOT NULL COMMENT '下单时间',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '最终合计价格，单位：元 totalPrice = totalProductPrice + totalTaxPrice - discountPrice',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品总金额',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额总金额',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠率',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `deposit_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '定金金额',

    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `out_count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已出库数量',
    `return_count`        decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已退货数量',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售订单';

-- ----------------------------
-- Table structure for erp_sale_order_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_order_item`;
CREATE TABLE `erp_sale_order_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '订单项主键',
    `order_id`        bigint         NOT NULL COMMENT '销售订单编号 关联 erp_sale_order.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总价',
    `tax_percent`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率',
    `tax_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额',
    `out_count`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已出库数量',
    `return_count`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已退货数量',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_order_id` (`order_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售订单项';

-- ----------------------------
-- Table structure for erp_sale_out
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_out`;
CREATE TABLE `erp_sale_out`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '出库单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '出库单号',
    `status`              int            NOT NULL COMMENT '审核状态 枚举 10:未审核,20:已审核',
    `customer_id`         bigint         NOT NULL COMMENT '客户编号 关联 erp_customer.id',
    `account_id`          bigint         NOT NULL COMMENT '账户编号 冗余 erp_customer.account_id',
    `sale_user_id`        bigint         NOT NULL COMMENT '销售员编号 冗余 erp_customer.sale_user_id',
    `out_time`            datetime       NOT NULL COMMENT '出库时间',
    `order_id`            bigint         NOT NULL COMMENT '销售订单编号 冗余 erp_sale_order.id',
    `order_no`            varchar(64)    NOT NULL COMMENT '销售订单号 冗余 erp_sale_order.no',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计金额',
    `receipt_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '已收款金额',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品总金额',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额总金额',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠率',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `other_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '其他金额',
    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售出库单';

-- ----------------------------
-- Table structure for erp_sale_out_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_out_item`;
CREATE TABLE `erp_sale_out_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '出库单项主键',
    `out_id`          bigint         NOT NULL COMMENT '出库单编号 关联 erp_sale_out.id',
    `order_item_id`   bigint         NOT NULL COMMENT '销售订单项编号 关联 erp_sale_order_item.id',
    `warehouse_id`    bigint         NOT NULL COMMENT '仓库编号 关联 erp_warehouse.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总价',
    `total_percent`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率',
    `tax_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_out_id` (`out_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售出库单项';

-- ----------------------------
-- Table structure for erp_sale_return
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_return`;
CREATE TABLE `erp_sale_return`
(
    `id`                  bigint         NOT NULL AUTO_INCREMENT COMMENT '退货单主键',
    `no`                  varchar(64)    NOT NULL COMMENT '退货单号',
    `status`              int            NOT NULL COMMENT '审核状态 枚举 10:未审核,20:已审核',
    `customer_id`         bigint         NOT NULL COMMENT '客户编号 关联 erp_customer.id',
    `account_id`          bigint         NOT NULL COMMENT '账户编号 冗余 erp_customer.account_id',
    `sale_user_id`        bigint         NOT NULL COMMENT '销售员编号 冗余 erp_customer.sale_user_id',
    `return_time`         datetime       NOT NULL COMMENT '退货时间',
    `order_id`            bigint         NOT NULL COMMENT '销售订单编号 冗余 erp_sale_order.id',
    `order_no`            varchar(64)    NOT NULL COMMENT '销售订单号 冗余 erp_sale_order.no',
    `total_count`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '合计数量',
    `total_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '最终合计价格，单位：元 totalPrice = totalProductPrice + totalTaxPrice - discountPrice',
    `refund_price`        decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '退款金额',
    `total_product_price` decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品总金额',
    `total_tax_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额总金额',
    `discount_percent`    decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠率',
    `discount_price`      decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '优惠金额',
    `other_price`         decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '其他金额',
    `remark`              varchar(512)            DEFAULT '' COMMENT '备注',
    `file_url`            varchar(1024)           DEFAULT '' COMMENT '附件地址',
    `creator`             varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`         datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_customer_id` (`customer_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售退货单';

-- ----------------------------
-- Table structure for erp_sale_return_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_sale_return_item`;
CREATE TABLE `erp_sale_return_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '退货单项主键',
    `return_id`       bigint         NOT NULL COMMENT '退货单编号 关联 erp_sale_return.id',
    `order_item_id`   bigint         NOT NULL COMMENT '销售订单单项编号 冗余 erp_sale_order_item.id',
    `warehouse_id`    bigint         NOT NULL COMMENT '仓库编号 关联 erp_warehouse.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '总价',
    `tax_percent`     decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税率',
    `tax_price`       decimal(10, 2) NOT NULL DEFAULT '0.00' COMMENT '税额',
    `remark`          varchar(512)            DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_return_id` (`return_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 销售退货单项';

-- ----------------------------
-- Table structure for erp_stock_check
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_check`;
CREATE TABLE `erp_stock_check`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '盘点编号',
    `no`          varchar(64)    NOT NULL DEFAULT '' COMMENT '盘点单号',
    `check_time`  datetime       NOT NULL COMMENT '盘点时间',
    `total_count` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计数量',
    `total_price` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `status`      tinyint        NOT NULL DEFAULT 0 COMMENT '状态: 0-待审核, 1-已审核, 2-已驳回',
    `remark`      varchar(512)   NULL COMMENT '备注',
    `file_url`    varchar(1024)  NULL     DEFAULT NULL COMMENT '附件 URL',
    `creator`     varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 库存盘点单';

-- ----------------------------
-- Table structure for erp_stock_check_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_check_item`;
CREATE TABLE `erp_stock_check_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '盘点项编号',
    `check_id`        bigint         NOT NULL DEFAULT 0 COMMENT '盘点编号',
    `warehouse_id`    bigint         NOT NULL DEFAULT 0 COMMENT '仓库编号',
    `product_id`      bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_unit_id` bigint         NOT NULL DEFAULT 0 COMMENT '产品单位编号',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品单价',
    `stock_count`     decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '账面数量（当前库存）',
    `actual_count`    decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '实际数量（实际库存）',
    `count`           decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '盈亏数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `remark`          varchar(512)   NULL COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_check_id` (`check_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 库存盘点单项';

-- ----------------------------
-- Table structure for erp_stock
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock`;
CREATE TABLE `erp_stock`
(
    `id`           bigint         NOT NULL AUTO_INCREMENT COMMENT '编号',
    `product_id`   bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `warehouse_id` bigint         NOT NULL DEFAULT 0 COMMENT '仓库编号',
    `count`        decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '库存数量',
    `creator`      varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`  datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 产品库存表';

-- ----------------------------
-- Table structure for erp_stock_in
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_in`;
CREATE TABLE `erp_stock_in`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '入库编号',
    `no`          varchar(64)    NOT NULL DEFAULT '' COMMENT '入库单号',
    `supplier_id` bigint         NOT NULL DEFAULT 0 COMMENT '供应商编号',
    `in_time`     datetime       NOT NULL COMMENT '入库时间',
    `total_count` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计数量',
    `total_price` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `status`      tinyint        NOT NULL DEFAULT 0 COMMENT '状态: 0-待审核, 1-已审核, 2-已驳回',
    `remark`      varchar(512)   NULL COMMENT '备注',
    `file_url`    varchar(1024)  NULL     DEFAULT NULL COMMENT '附件 URL',
    `creator`     varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 其它入库单';

-- ----------------------------
-- Table structure for erp_stock_in_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_in_item`;
CREATE TABLE `erp_stock_in_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '入库项编号',
    `in_id`           bigint         NOT NULL DEFAULT 0 COMMENT '入库编号',
    `warehouse_id`    bigint         NOT NULL DEFAULT 0 COMMENT '仓库编号',
    `product_id`      bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_unit_id` bigint         NOT NULL DEFAULT 0 COMMENT '产品单位编号',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `remark`          varchar(512)   NULL COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_in_id` (`in_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 其它入库单项';

-- ----------------------------
-- Table structure for erp_stock_move
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_move`;
CREATE TABLE `erp_stock_move`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '调拨编号',
    `no`          varchar(64)    NOT NULL DEFAULT '' COMMENT '调拨单号',
    `move_time`   datetime       NOT NULL COMMENT '调拨时间',
    `total_count` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计数量',
    `total_price` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `status`      tinyint        NOT NULL DEFAULT 0 COMMENT '状态: 0-待审核, 1-已审核, 2-已驳回',
    `remark`      varchar(512)   NULL COMMENT '备注',
    `file_url`    varchar(1024)  NULL     DEFAULT NULL COMMENT '附件 URL',
    `creator`     varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 库存调拨单';

-- ----------------------------
-- Table structure for erp_stock_move_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_move_item`;
CREATE TABLE `erp_stock_move_item`
(
    `id`                bigint         NOT NULL AUTO_INCREMENT COMMENT '调拨项编号',
    `move_id`           bigint         NOT NULL DEFAULT 0 COMMENT '调拨编号',
    `from_warehouse_id` bigint         NOT NULL DEFAULT 0 COMMENT '调出仓库编号',
    `to_warehouse_id`   bigint         NOT NULL DEFAULT 0 COMMENT '调入仓库编号',
    `product_id`        bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_unit_id`   bigint         NOT NULL DEFAULT 0 COMMENT '产品单位编号',
    `product_price`     decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品单价',
    `count`             decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品数量',
    `total_price`       decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `remark`            varchar(512)   NULL COMMENT '备注',
    `creator`           varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`       datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`       datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_move_id` (`move_id` ASC) USING BTREE,
    INDEX `idx_from_warehouse_id` (`from_warehouse_id` ASC) USING BTREE,
    INDEX `idx_to_warehouse_id` (`to_warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 库存调拨单项';

-- ----------------------------
-- Table structure for erp_stock_out
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_out`;
CREATE TABLE `erp_stock_out`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '出库编号',
    `no`          varchar(64)    NOT NULL DEFAULT '' COMMENT '出库单号',
    `customer_id` bigint         NOT NULL DEFAULT 0 COMMENT '客户编号',
    `out_time`    datetime       NOT NULL COMMENT '出库时间',
    `total_count` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计数量',
    `total_price` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `status`      tinyint        NOT NULL DEFAULT 0 COMMENT '状态: 0-待审核, 1-已审核, 2-已驳回',
    `remark`      varchar(512)   NULL COMMENT '备注',
    `file_url`    varchar(1024)  NULL     DEFAULT NULL COMMENT '附件 URL',
    `creator`     varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 其它出库单';

-- ----------------------------
-- Table structure for erp_stock_out_item
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_out_item`;
CREATE TABLE `erp_stock_out_item`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '出库单项主键',
    `out_id`          bigint         NOT NULL COMMENT '出库单编号 关联 erp_stock_out.id',
    `warehouse_id`    bigint         NOT NULL COMMENT '仓库编号 关联 erp_warehouse.id',
    `product_id`      bigint         NOT NULL COMMENT '产品编号 关联 erp_product.id',
    `product_unit_id` bigint         NOT NULL COMMENT '产品单位编号 冗余 erp_product.unit_id',
    `product_price`   decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品单价',
    `count`           decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '产品数量',
    `total_price`     decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '合计金额，单位：元',
    `remark`          varchar(512)   NULL COMMENT '备注',
    `creator`         varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_out_id` (`out_id` ASC) USING BTREE,
    INDEX `idx_warehouse_id` (`warehouse_id` ASC) USING BTREE,
    INDEX `idx_product_id` (`product_id` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 其它出库单项';

-- ----------------------------
-- Table structure for erp_stock_record
-- ----------------------------
DROP TABLE IF EXISTS `erp_stock_record`;
CREATE TABLE `erp_stock_record`
(
    `id`          bigint         NOT NULL AUTO_INCREMENT COMMENT '编号',
    `productId`   bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `warehouseId` bigint         NOT NULL DEFAULT 0 COMMENT '仓库编号',
    `count`       decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '出入库数量: 正数表示入库，负数表示出库',
    `total_count` decimal(10, 2) NOT NULL DEFAULT 0.00 COMMENT '总库存量',
    `biz_type`    tinyint        NOT NULL DEFAULT 0 COMMENT '业务类型: 1-入库, 2-出库, 3-调拨, 4-盘点',
    `biz_id`      bigint         NOT NULL DEFAULT 0 COMMENT '业务编号',
    `biz_item_id` bigint         NOT NULL DEFAULT 0 COMMENT '业务项编号',
    `biz_no`      varchar(64)    NOT NULL DEFAULT '' COMMENT '业务单号',
    `creator`     varchar(64)             DEFAULT '' COMMENT '创建者',
    `create_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)             DEFAULT '' COMMENT '更新者',
    `update_time` datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 库存明细';


-- ----------------------------
-- Table structure for erp_warehouse
-- ----------------------------
DROP TABLE IF EXISTS `erp_warehouse`;
CREATE TABLE `erp_warehouse`
(
    `id`              bigint       NOT NULL AUTO_INCREMENT COMMENT '仓库主键',
    `name`            varchar(255) NOT NULL COMMENT '仓库名称',
    `address`         varchar(255)          DEFAULT '' COMMENT '仓库地址',
    `sort`            int          NOT NULL DEFAULT '0' COMMENT '排序',
    `principal`       varchar(64)           DEFAULT '' COMMENT '负责人',
    `warehouse_price` decimal(10, 2)        DEFAULT '0.00' COMMENT '仓储费，单位：元',
    `truckage_price`  decimal(10, 2)        DEFAULT '0.00' COMMENT '搬运费，单位：元',
    `status`          int          NOT NULL DEFAULT '0' COMMENT '开启状态 枚举 0:开启,1:关闭',
    `default_status`  bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否默认',
    `remark`          varchar(512)          DEFAULT '' COMMENT '备注',
    `creator`         varchar(64)           DEFAULT '' COMMENT '创建者',
    `create_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)           DEFAULT '' COMMENT '更新者',
    `update_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_status` (`status` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'ERP 仓库';
