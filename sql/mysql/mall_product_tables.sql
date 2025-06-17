CREATE DATABASE IF NOT EXISTS `pei_mall_product`;
USE `pei_mall_product`;

-- ----------------------------
-- Table structure for product_brand
-- ----------------------------
DROP TABLE IF EXISTS `product_brand`;
CREATE TABLE `product_brand`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '品牌编号',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '品牌名称',
    `pic_url`     varchar(512)  NOT NULL DEFAULT '' COMMENT '品牌图片',
    `sort`        int           NOT NULL DEFAULT 0 COMMENT '品牌排序',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '品牌描述',
    `status`      tinyint       NOT NULL DEFAULT 0 COMMENT '状态: 0-禁用, 1-启用',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品品牌表';


-- ----------------------------
-- Table structure for product_category
-- ----------------------------
DROP TABLE IF EXISTS `product_category`;
CREATE TABLE `product_category`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '分类编号',
    `parent_id`   bigint       NOT NULL DEFAULT 0 COMMENT '父分类编号: 0表示根分类',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '分类名称',
    `pic_url`     varchar(512) NOT NULL DEFAULT '' COMMENT '移动端分类图 (建议180*180分辨率)',
    `sort`        int          NOT NULL DEFAULT 0 COMMENT '分类排序',
    `status`      tinyint      NOT NULL DEFAULT 0 COMMENT '开启状态: 0-禁用, 1-启用',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_parent_id` (`parent_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品分类表';

-- ----------------------------
-- Table structure for product_comment
-- ----------------------------
DROP TABLE IF EXISTS `product_comment`;
CREATE TABLE `product_comment`
(
    `id`                 bigint       NOT NULL AUTO_INCREMENT COMMENT '评论编号，主键自增',
    `user_id`            bigint       NOT NULL DEFAULT 0 COMMENT '评价人的用户编号',
    `user_nickname`      varchar(255) NOT NULL DEFAULT '' COMMENT '评价人名称',
    `user_avatar`        varchar(512) NOT NULL DEFAULT '' COMMENT '评价人头像',
    `anonymous`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否匿名: 0-否, 1-是',
    `order_id`           bigint       NOT NULL DEFAULT 0 COMMENT '交易订单编号',
    `order_item_id`      bigint       NOT NULL DEFAULT 0 COMMENT '交易订单项编号',
    `spu_id`             bigint       NOT NULL DEFAULT 0 COMMENT '商品SPU编号',
    `spu_name`           varchar(255) NOT NULL DEFAULT '' COMMENT '商品SPU名称',
    `sku_id`             bigint       NOT NULL DEFAULT 0 COMMENT '商品SKU编号',
    `sku_pic_url`        varchar(512) NOT NULL DEFAULT '' COMMENT '商品SKU图片地址',
    `sku_properties`     text         NOT NULL COMMENT '属性数组，JSON格式',
    `visible`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否可见: 0-隐藏, 1-显示',
    `scores`             tinyint      NOT NULL DEFAULT 0 COMMENT '评分星级: 1-5分',
    `description_scores` tinyint      NOT NULL DEFAULT 0 COMMENT '描述星级: 1-5星',
    `benefit_scores`     tinyint      NOT NULL DEFAULT 0 COMMENT '服务星级: 1-5星',
    `content`            text         NULL COMMENT '评论内容',
    `pic_urls`           text         NOT NULL COMMENT '评论图片地址数组',
    `reply_status`       bit(1)       NOT NULL DEFAULT b'0' COMMENT '商家是否回复: 0-未回复, 1-已回复',
    `reply_user_id`      bigint       NOT NULL DEFAULT 0 COMMENT '回复管理员编号',
    `reply_content`      text         NULL COMMENT '商家回复内容',
    `reply_time`         datetime     NULL     DEFAULT NULL COMMENT '商家回复时间',
    `creator`            varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_spu_id` (`spu_id`),
    INDEX `idx_sku_id` (`sku_id`),
    INDEX `idx_user_id` (`user_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品评论表';

-- ----------------------------
-- Table structure for product_favorite
-- ----------------------------
DROP TABLE IF EXISTS `product_favorite`;
CREATE TABLE `product_favorite`
(
    `id`          bigint      NOT NULL AUTO_INCREMENT COMMENT '收藏编号',
    `user_id`     bigint      NOT NULL DEFAULT 0 COMMENT '用户编号: 关联 MemberUserDO 的 id 编号',
    `spu_id`      bigint      NOT NULL DEFAULT 0 COMMENT '商品 SPU 编号: 关联 product_spu 表的 id 编号',
    `creator`     varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`),
    INDEX `idx_spu_id` (`spu_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品收藏表';

-- ----------------------------
-- Table structure for product_browse_history
-- ----------------------------
DROP TABLE IF EXISTS `product_browse_history`;
CREATE TABLE `product_browse_history`
(
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '记录编号',
    `spu_id`       bigint      NOT NULL DEFAULT 0 COMMENT '商品 SPU 编号',
    `user_id`      bigint      NOT NULL DEFAULT 0 COMMENT '用户编号',
    `user_deleted` bit(1)      NOT NULL DEFAULT b'0' COMMENT '用户是否删除',
    `creator`      varchar(64) NULL     DEFAULT '' COMMENT '创建者',
    `create_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64) NULL     DEFAULT '' COMMENT '更新者',
    `update_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`),
    INDEX `idx_spu_id` (`spu_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品浏览记录表';

-- ----------------------------
-- Table structure for product_property
-- ----------------------------
DROP TABLE IF EXISTS `product_property`;
CREATE TABLE `product_property`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '名称',
    `remark`      varchar(255) NULL     DEFAULT '' COMMENT '备注',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品属性项表';

-- ----------------------------
-- Table structure for product_property_value
-- ----------------------------
DROP TABLE IF EXISTS `product_property_value`;
CREATE TABLE `product_property_value`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `property_id` bigint       NOT NULL DEFAULT 0 COMMENT '属性项的编号: 关联 product_property 表的 id 编号',
    `name`        varchar(255) NOT NULL DEFAULT '' COMMENT '名称',
    `remark`      varchar(255) NULL     DEFAULT '' COMMENT '备注',
    `creator`     varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_property_id` (`property_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品属性值表';

-- ----------------------------
-- Table structure for product_sku
-- ----------------------------
DROP TABLE IF EXISTS `product_sku`;
CREATE TABLE `product_sku`
(
    `id`                     bigint       NOT NULL AUTO_INCREMENT COMMENT '商品 SKU 编号',
    `spu_id`                 bigint       NOT NULL DEFAULT 0 COMMENT 'SPU 编号: 关联 product_spu 表的 id 编号',
    `properties`             text         NULL COMMENT '商品属性：JSON 格式',
    `price`                  int          NOT NULL DEFAULT 0 COMMENT '商品价格，单位：分',
    `market_price`           int          NOT NULL DEFAULT 0 COMMENT '市场价，单位：分',
    `cost_price`             int          NOT NULL DEFAULT 0 COMMENT '成本价，单位：分',
    `bar_code`               varchar(64)  NOT NULL DEFAULT '' COMMENT '商品条码',
    `pic_url`                varchar(512) NOT NULL DEFAULT '' COMMENT '图片地址',
    `stock`                  int          NOT NULL DEFAULT 0 COMMENT '库存',
    `weight`                 double       NOT NULL DEFAULT 0 COMMENT '商品重量，单位：kg 千克',
    `volume`                 double       NOT NULL DEFAULT 0 COMMENT '商品体积，单位：m^3 平米',
    `first_brokerage_price`  int          NOT NULL DEFAULT 0 COMMENT '一级分销的佣金，单位：分',
    `second_brokerage_price` int          NOT NULL DEFAULT 0 COMMENT '二级分销的佣金，单位：分',
    `sales_count`            int          NOT NULL DEFAULT 0 COMMENT '商品销量',
    `creator`                varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`            datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`            datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`),
    INDEX `idx_spu_id` (`spu_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='商品 SKU 表';

-- ----------------------------
-- Table structure for product_spu
-- ----------------------------
DROP TABLE IF EXISTS `product_spu`;
CREATE TABLE `product_spu`
(
    `id`                   BIGINT       NOT NULL AUTO_INCREMENT COMMENT '商品 SPU 编号',

    -- ========== 基本信息 ==========
    `name`                 VARCHAR(255) NOT NULL DEFAULT '' COMMENT '商品名称',
    `keyword`              VARCHAR(255) NOT NULL DEFAULT '' COMMENT '关键字',
    `introduction`         VARCHAR(512) NOT NULL DEFAULT '' COMMENT '商品简介',
    `description`          TEXT         NULL COMMENT '商品详情',
    `category_id`          BIGINT       NOT NULL DEFAULT 0 COMMENT '商品分类编号: 关联 product_category.id',
    `brand_id`             BIGINT       NOT NULL DEFAULT 0 COMMENT '商品品牌编号: 关联 product_brand.id',
    `pic_url`              VARCHAR(512) NOT NULL DEFAULT '' COMMENT '商品封面图',
    `slider_pic_urls`      TEXT         NULL COMMENT '商品轮播图: JSON 格式',

    `sort`                 INT          NOT NULL DEFAULT 0 COMMENT '排序字段',
    `status`               INT          NOT NULL DEFAULT 0 COMMENT '商品状态: 枚举 ProductSpuStatusEnum',

    -- ========== SKU 相关字段 ==========
    `spec_type`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '规格类型: false - 单规格, true - 多规格',
    `price`                INT          NOT NULL DEFAULT 0 COMMENT '商品价格，单位：分',
    `market_price`         INT          NOT NULL DEFAULT 0 COMMENT '市场价，单位：分',
    `cost_price`           INT          NOT NULL DEFAULT 0 COMMENT '成本价，单位：分',
    `stock`                INT          NOT NULL DEFAULT 0 COMMENT '库存',

    -- ========== 物流相关字段 ==========
    `delivery_types`       TEXT         NULL COMMENT '配送方式数组: JSON 格式',
    `delivery_template_id` BIGINT       NOT NULL DEFAULT 0 COMMENT '物流配置模板编号',

    -- ========== 营销相关字段 ==========
    `give_integral`        INT          NOT NULL DEFAULT 0 COMMENT '赠送积分',
    `sub_commission_type`  bit(1)       NOT NULL DEFAULT b'0' COMMENT '分销类型: false - 默认, true - 自行设置',

    -- ========== 统计相关字段 ==========
    `sales_count`          INT          NOT NULL DEFAULT 0 COMMENT '商品销量',
    `virtual_sales_count`  INT          NOT NULL DEFAULT 0 COMMENT '虚拟销量',
    `browse_count`         INT          NOT NULL DEFAULT 0 COMMENT '浏览量',

    -- BaseDO 公共字段
    `creator`              VARCHAR(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              VARCHAR(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              BIT(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',

    PRIMARY KEY (`id`),
    INDEX `idx_category_id` (`category_id`),
    INDEX `idx_brand_id` (`brand_id`)
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
    COMMENT = '商品 SPU 表';
