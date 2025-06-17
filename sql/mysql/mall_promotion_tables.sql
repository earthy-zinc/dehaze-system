CREATE DATABASE IF NOT EXISTS `pei_mall_promotion`;
USE `pei_mall_promotion`;

-- ----------------------------
-- Table structure for promotion_article_category
-- ----------------------------
DROP TABLE IF EXISTS `promotion_article_category`;
CREATE TABLE `promotion_article_category`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '文章分类编号',
    `name`        varchar(255)  NOT NULL COMMENT '文章分类名称',
    `pic_url`     varchar(1024) NOT NULL COMMENT '图标地址',
    `status`      tinyint       NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `sort`        int           NOT NULL COMMENT '排序',

    `creator`     VARCHAR(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     VARCHAR(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     BIT(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',

    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '文章分类表';

-- ----------------------------
-- Table structure for promotion_article
-- ----------------------------
DROP TABLE IF EXISTS `promotion_article`;
CREATE TABLE `promotion_article`
(
    `id`               bigint        NOT NULL AUTO_INCREMENT COMMENT '文章管理编号',
    `category_id`      bigint        NOT NULL COMMENT '分类编号',
    `spu_id`           bigint        NOT NULL COMMENT '关联商品编号',
    `title`            varchar(255)  NOT NULL COMMENT '文章标题',
    `author`           varchar(64)   NOT NULL COMMENT '文章作者',
    `pic_url`          varchar(1024) NOT NULL COMMENT '文章封面图片地址',
    `introduction`     varchar(512)  NOT NULL COMMENT '文章简介',
    `browse_count`     int           NOT NULL DEFAULT 0 COMMENT '浏览次数',
    `sort`             int           NOT NULL COMMENT '排序',
    `status`           tinyint       NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `recommend_hot`    bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否热门: 0-否, 1-是',
    `recommend_banner` bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否轮播图: 0-否, 1-是',
    `content`          text          NULL COMMENT '文章内容',
    `creator`          varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '文章管理表';

-- ----------------------------
-- Table structure for promotion_banner
-- ----------------------------
DROP TABLE IF EXISTS `promotion_banner`;
CREATE TABLE `promotion_banner`
(
    `id`           bigint        NOT NULL AUTO_INCREMENT COMMENT '编号',
    `title`        varchar(255)  NOT NULL COMMENT '标题',
    `url`          varchar(1024) NOT NULL COMMENT '跳转链接',
    `pic_url`      varchar(1024) NOT NULL COMMENT '图片链接',
    `sort`         int           NOT NULL COMMENT '排序',
    `status`       tinyint       NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `position`     tinyint       NOT NULL COMMENT '定位: 1-首页, 2-秒杀活动页, 3-砍价活动页, 4-限时折扣页, 5-满减送页',
    `memo`         varchar(512)  NOT NULL COMMENT '备注',
    `browse_count` int           NOT NULL DEFAULT 0 COMMENT '点击次数',
    `creator`      varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'Banner表';

-- ----------------------------
-- Table structure for promotion_bargain_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_bargain_activity`;
CREATE TABLE `promotion_bargain_activity`
(
    `id`                  bigint       NOT NULL AUTO_INCREMENT COMMENT '砍价活动编号',
    `name`                varchar(255) NOT NULL COMMENT '砍价活动名称',
    `start_time`          datetime     NOT NULL COMMENT '活动开始时间',
    `end_time`            datetime     NOT NULL COMMENT '活动结束时间',
    `status`              tinyint      NOT NULL COMMENT '活动状态: 0-开启, 1-禁用',
    `spu_id`              bigint       NOT NULL COMMENT '商品SPU编号',
    `sku_id`              bigint       NOT NULL COMMENT '商品SKU编号',
    `bargain_first_price` int          NOT NULL COMMENT '砍价起始价格，单位：分',
    `bargain_min_price`   int          NOT NULL COMMENT '砍价底价，单位：分',
    `stock`               int          NOT NULL COMMENT '砍价库存(剩余库存砍价时扣减)',
    `total_stock`         int          NOT NULL COMMENT '砍价总库存',
    `help_max_count`      int          NOT NULL COMMENT '砍价人数',
    `bargain_count`       int          NOT NULL COMMENT '帮砍次数',
    `total_limit_count`   int          NOT NULL COMMENT '总限购数量',
    `random_min_price`    int          NOT NULL COMMENT '用户每次砍价的最小金额，单位：分',
    `random_max_price`    int          NOT NULL COMMENT '用户每次砍价的最大金额，单位：分',
    `creator`             varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '砍价活动表';


-- ----------------------------
-- Table structure for promotion_bargain_help
-- ----------------------------
DROP TABLE IF EXISTS `promotion_bargain_help`;
CREATE TABLE `promotion_bargain_help`
(
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `activity_id`  bigint      NOT NULL COMMENT '砍价活动编号',
    `record_id`    bigint      NOT NULL COMMENT '砍价记录编号',
    `user_id`      bigint      NOT NULL COMMENT '用户编号',
    `reduce_price` int         NOT NULL COMMENT '减少价格，单位：分',
    `creator`      varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`  datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '砍价助力表';

-- ----------------------------
-- Table structure for promotion_bargain_record
-- ----------------------------
DROP TABLE IF EXISTS `promotion_bargain_record`;
CREATE TABLE `promotion_bargain_record`
(
    `id`                  bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `user_id`             bigint      NOT NULL COMMENT '用户编号',
    `activity_id`         bigint      NOT NULL COMMENT '砍价活动编号',
    `spu_id`              bigint      NOT NULL COMMENT '商品SPU编号',
    `sku_id`              bigint      NOT NULL COMMENT '商品SKU编号',
    `bargain_first_price` int         NOT NULL COMMENT '砍价起始价格，单位：分',
    `bargain_price`       int         NOT NULL COMMENT '当前砍价，单位：分',
    `status`              tinyint     NOT NULL COMMENT '砍价状态: 1-砍价中, 2-砍价成功, 3-砍价失败',
    `end_time`            datetime    NOT NULL COMMENT '结束时间',
    `order_id`            bigint      NOT NULL COMMENT '订单编号',
    `creator`             varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '砍价记录表';

-- ----------------------------
-- Table structure for promotion_combination_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_combination_activity`;
CREATE TABLE `promotion_combination_activity`
(
    `id`                 bigint       NOT NULL AUTO_INCREMENT COMMENT '活动编号',
    `name`               varchar(255) NOT NULL COMMENT '拼团名称',
    `spu_id`             bigint       NOT NULL COMMENT '商品SPU编号',
    `total_limit_count`  int          NOT NULL COMMENT '总限购数量',
    `single_limit_count` int          NOT NULL COMMENT '单次限购数量',
    `start_time`         datetime     NOT NULL COMMENT '开始时间',
    `end_time`           datetime     NOT NULL COMMENT '结束时间',
    `user_size`          int          NOT NULL COMMENT '几人团',
    `virtual_group`      bit(1)       NOT NULL DEFAULT b'0' COMMENT '虚拟成团: 0-否, 1-是',
    `status`             tinyint      NOT NULL COMMENT '活动状态: 0-开启, 1-禁用',
    `limit_duration`     int          NOT NULL COMMENT '限制时长（小时）',
    `creator`            varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '拼团活动表';

-- ----------------------------
-- Table structure for promotion_combination_product
-- ----------------------------
DROP TABLE IF EXISTS `promotion_combination_product`;
CREATE TABLE `promotion_combination_product`
(
    `id`                  bigint      NOT NULL AUTO_INCREMENT COMMENT '编号',
    `activity_id`         bigint      NOT NULL COMMENT '拼团活动编号',
    `spu_id`              bigint      NOT NULL COMMENT '商品SPU编号',
    `sku_id`              bigint      NOT NULL COMMENT '商品SKU编号',
    `combination_price`   int         NOT NULL COMMENT '拼团价格，单位分',
    `activity_status`     tinyint     NOT NULL COMMENT '拼团商品状态',
    `activity_start_time` datetime    NOT NULL COMMENT '活动开始时间点',
    `activity_end_time`   datetime    NOT NULL COMMENT '活动结束时间点',
    `creator`             varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '拼团商品表';

-- ----------------------------
-- Table structure for promotion_combination_record
-- ----------------------------
DROP TABLE IF EXISTS `promotion_combination_record`;
CREATE TABLE `promotion_combination_record`
(
    `id`                bigint        NOT NULL AUTO_INCREMENT COMMENT '编号',
    `activity_id`       bigint        NOT NULL COMMENT '拼团活动编号',
    `combination_price` int           NOT NULL COMMENT '拼团商品单价',
    `spu_id`            bigint        NOT NULL COMMENT 'SPU编号',
    `spu_name`          varchar(255)  NOT NULL COMMENT '商品名字',
    `pic_url`           varchar(1024) NOT NULL COMMENT '商品图片',
    `sku_id`            bigint        NOT NULL COMMENT 'SKU编号',
    `count`             int           NOT NULL COMMENT '购买的商品数量',
    `userId`            bigint        NOT NULL COMMENT '用户编号',
    `nickname`          varchar(64)   NOT NULL COMMENT '用户昵称',
    `avatar`            varchar(1024) NOT NULL COMMENT '用户头像',
    `head_id`           bigint        NOT NULL COMMENT '团长编号: 0-团长',
    `status`            tinyint       NOT NULL COMMENT '开团状态: 0-进行中, 1-拼团成功, 2-拼团失败',
    `order_id`          bigint        NOT NULL COMMENT '订单编号',
    `user_size`         int           NOT NULL COMMENT '开团需要人数',
    `user_count`        int           NOT NULL COMMENT '已加入拼团人数',
    `virtual_group`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否虚拟成团: 0-否, 1-是',
    `expire_time`       datetime      NOT NULL COMMENT '过期时间',
    `start_time`        datetime      NOT NULL COMMENT '开始时间',
    `end_time`          datetime      NOT NULL COMMENT '结束时间',
    `creator`           varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`           varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`       datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`           bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '拼团记录表';

-- ----------------------------
-- Table structure for promotion_coupon
-- ----------------------------
DROP TABLE IF EXISTS `promotion_coupon`;
CREATE TABLE `promotion_coupon`
(
    `id`                   bigint       NOT NULL AUTO_INCREMENT COMMENT '优惠劵编号',
    `template_id`          bigint       NOT NULL COMMENT '优惠劵模板编号',
    `name`                 varchar(255) NOT NULL COMMENT '优惠劵名',
    `status`               tinyint      NOT NULL COMMENT '优惠码状态: 1-未使用, 2-已使用, 3-已过期',
    `user_id`              bigint       NOT NULL COMMENT '用户编号',
    `take_type`            tinyint      NOT NULL COMMENT '领取类型: 1-直接领取, 2-指定发放, 3-新人券',
    `use_price`            int          NOT NULL COMMENT '是否设置满多少金额可用',
    `valid_start_time`     datetime     NOT NULL COMMENT '生效开始时间',
    `valid_end_time`       datetime     NOT NULL COMMENT '生效结束时间',
    `product_scope`        tinyint      NOT NULL COMMENT '商品范围: 1-全部商品, 2-指定商品, 3-指定品类',
    `product_scope_values` text         NULL COMMENT '商品范围编号的数组',
    `discount_type`        tinyint      NOT NULL COMMENT '折扣类型: 1-满减, 2-折扣',
    `discount_percent`     int          NOT NULL COMMENT '折扣百分比',
    `discount_price`       int          NOT NULL COMMENT '优惠金额，单位：分',
    `discount_limit_price` int          NOT NULL COMMENT '折扣上限',
    `use_order_id`         bigint       NOT NULL COMMENT '使用订单号',
    `use_time`             datetime     NOT NULL COMMENT '使用时间',
    `creator`              varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '优惠券表';

-- ----------------------------
-- Table structure for promotion_coupon_template
-- ----------------------------
DROP TABLE IF EXISTS `promotion_coupon_template`;
CREATE TABLE `promotion_coupon_template`
(
    `id`                   bigint        NOT NULL AUTO_INCREMENT COMMENT '模板编号',
    `name`                 varchar(255)  NOT NULL COMMENT '优惠劵名',
    `description`          varchar(1024) NOT NULL COMMENT '优惠券说明',
    `status`               tinyint       NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `total_count`          int           NOT NULL COMMENT '发放数量: -1-不限制领取数量',
    `take_limit_count`     int           NOT NULL COMMENT '每人限领个数: -1-不限制',
    `take_type`            tinyint       NOT NULL COMMENT '领取方式: 1-直接领取, 2-指定发放, 3-新人券',
    `use_price`            int           NOT NULL COMMENT '满多少金额可用，单位：分: 0-不限制',
    `product_scope`        tinyint       NOT NULL COMMENT '商品范围: 1-全部商品, 2-指定商品, 3-指定品类',
    `product_scope_values` text          NOT NULL COMMENT '商品范围编号的数组',
    `validity_type`        tinyint       NOT NULL COMMENT '生效日期类型: 1-固定日期, 2-领取之后',
    `valid_start_time`     datetime      NOT NULL COMMENT '生效开始时间',
    `valid_end_time`       datetime      NOT NULL COMMENT '生效结束时间',
    `fixed_start_term`     int           NOT NULL COMMENT '领取日期-开始天数',
    `fixed_end_term`       int           NOT NULL COMMENT '领取日期-结束天数',
    `discount_type`        tinyint       NOT NULL COMMENT '折扣类型: 1-满减, 2-折扣',
    `discount_percent`     int           NOT NULL COMMENT '折扣百分比',
    `discount_price`       int           NOT NULL COMMENT '优惠金额，单位：分',
    `discount_limit_price` int           NOT NULL COMMENT '折扣上限，仅在折扣时生效',
    `take_count`           int           NOT NULL COMMENT '领取优惠券的数量',
    `use_count`            int           NOT NULL COMMENT '使用优惠券的次数',
    `creator`              varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`          datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '优惠券模板表';

-- ----------------------------
-- Table structure for promotion_discount_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_discount_activity`;
CREATE TABLE `promotion_discount_activity`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '活动编号',
    `name`        varchar(255) NOT NULL COMMENT '活动标题',
    `status`      tinyint      NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `start_time`  datetime     NOT NULL COMMENT '开始时间',
    `end_time`    datetime     NOT NULL COMMENT '结束时间',
    `remark`      varchar(512) NOT NULL COMMENT '备注',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '限时折扣活动表';

-- ----------------------------
-- Table structure for promotion_discount_product
-- ----------------------------
DROP TABLE IF EXISTS `promotion_discount_product`;
CREATE TABLE `promotion_discount_product`
(
    `id`                  bigint       NOT NULL AUTO_INCREMENT COMMENT '编号',
    `activity_id`         bigint       NOT NULL COMMENT '限时折扣活动编号',
    `spu_id`              bigint       NOT NULL COMMENT '商品SPU编号',
    `sku_id`              bigint       NOT NULL COMMENT '商品SKU编号',
    `discount_type`       tinyint      NOT NULL COMMENT '折扣类型: 1-满减, 2-折扣',
    `discount_percent`    int          NOT NULL COMMENT '折扣百分比',
    `discount_price`      int          NOT NULL COMMENT '优惠金额，单位：分',
    `activity_name`       varchar(255) NOT NULL COMMENT '活动标题',
    `activity_status`     tinyint      NOT NULL COMMENT '活动状态: 0-开启, 1-禁用',
    `activity_start_time` datetime     NOT NULL COMMENT '活动开始时间点',
    `activity_end_time`   datetime     NOT NULL COMMENT '活动结束时间点',
    `creator`             varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '限时折扣商品表';

-- ----------------------------
-- Table structure for promotion_diy_page
-- ----------------------------
DROP TABLE IF EXISTS `promotion_diy_page`;
CREATE TABLE `promotion_diy_page`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '装修页面编号',
    `template_id`      bigint       NOT NULL COMMENT '装修模板编号',
    `name`             varchar(255) NOT NULL COMMENT '页面名称',
    `remark`           varchar(512) NOT NULL COMMENT '备注',
    `preview_pic_urls` text         NOT NULL COMMENT '预览图列表',
    `property`         text         NOT NULL COMMENT '页面属性',
    `creator`          varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '装修页面表';

-- ----------------------------
-- Table structure for promotion_diy_template
-- ----------------------------
DROP TABLE IF EXISTS `promotion_diy_template`;
CREATE TABLE `promotion_diy_template`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT COMMENT '装修模板编号',
    `name`             varchar(255) NOT NULL COMMENT '模板名称',
    `used`             bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否使用: 0-否, 1-是',
    `used_time`        datetime     NOT NULL COMMENT '使用时间',
    `remark`           varchar(512) NOT NULL COMMENT '备注',
    `preview_pic_urls` text         NOT NULL COMMENT '预览图列表',
    `property`         text         NOT NULL COMMENT 'uni-app底部导航属性',
    `creator`          varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '装修模板表';

-- ----------------------------
-- Table structure for promotion_kefu_conversation
-- ----------------------------
DROP TABLE IF EXISTS `promotion_kefu_conversation`;
CREATE TABLE `promotion_kefu_conversation`
(
    `id`                         bigint        NOT NULL AUTO_INCREMENT COMMENT '会话编号',
    `user_id`                    bigint        NOT NULL COMMENT '会话所属用户编号',
    `last_message_time`          datetime      NOT NULL COMMENT '最后聊天时间',
    `last_message_content`       varchar(1024) NOT NULL COMMENT '最后聊天内容',
    `last_message_content_type`  tinyint       NOT NULL COMMENT '最后发送的消息类型: 1-文本消息, 2-图片消息, 3-语音消息, 4-视频消息, 5-系统消息, 10-商品消息, 11-订单消息',
    `admin_pinned`               bit(1)        NOT NULL DEFAULT b'0' COMMENT '管理端置顶: 0-否, 1-是',
    `user_deleted`               bit(1)        NOT NULL DEFAULT b'0' COMMENT '用户是否可见: 0-可见, 1-不可见',
    `admin_deleted`              bit(1)        NOT NULL DEFAULT b'0' COMMENT '管理员是否可见: 0-可见, 1-不可见',
    `admin_unread_message_count` int           NOT NULL COMMENT '管理员未读消息数',
    `creator`                    varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`                datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                    varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`                datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                    bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '客服会话表';

-- ----------------------------
-- Table structure for promotion_kefu_message
-- ----------------------------
DROP TABLE IF EXISTS `promotion_kefu_message`;
CREATE TABLE `promotion_kefu_message`
(
    `id`              bigint      NOT NULL AUTO_INCREMENT COMMENT '消息编号',
    `conversation_id` bigint      NOT NULL COMMENT '会话编号',
    `sender_id`       bigint      NOT NULL COMMENT '发送人编号',
    `sender_type`     tinyint     NOT NULL COMMENT '发送人类型: 1-用户, 2-管理员, 3-系统',
    `receiver_id`     bigint      NOT NULL COMMENT '接收人编号',
    `receiver_type`   tinyint     NOT NULL COMMENT '接收人类型: 1-用户, 2-管理员, 3-系统',
    `content_type`    tinyint     NOT NULL COMMENT '消息类型: 1-文本消息, 2-图片消息, 3-语音消息, 4-视频消息, 5-系统消息, 10-商品消息, 11-订单消息',
    `content`         text        NOT NULL COMMENT '消息内容',
    `read_status`     bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否已读: 0-未读, 1-已读',
    `creator`         varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '客服消息表';

-- ----------------------------
-- Table structure for promotion_point_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_point_activity`;
CREATE TABLE `promotion_point_activity`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '积分商城活动编号',
    `spu_id`      bigint       NOT NULL COMMENT '积分商城活动商品',
    `status`      tinyint      NOT NULL COMMENT '活动状态: 0-开启, 1-禁用',
    `remark`      varchar(512) NOT NULL COMMENT '备注',
    `sort`        int          NOT NULL COMMENT '排序',
    `stock`       int          NOT NULL COMMENT '积分商城活动库存(剩余库存积分兑换时扣减)',
    `total_stock` int          NOT NULL COMMENT '积分商城活动总库存',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '积分商城活动表';

-- ----------------------------
-- Table structure for promotion_point_product
-- ----------------------------
DROP TABLE IF EXISTS `promotion_point_product`;
CREATE TABLE `promotion_point_product`
(
    `id`              bigint      NOT NULL AUTO_INCREMENT COMMENT '积分商城商品编号',
    `activity_id`     bigint      NOT NULL COMMENT '积分商城活动编号',
    `spu_id`          bigint      NOT NULL COMMENT '商品SPU编号',
    `sku_id`          bigint      NOT NULL COMMENT '商品SKU编号',
    `count`           int         NOT NULL COMMENT '可兑换次数',
    `point`           int         NOT NULL COMMENT '所需兑换积分',
    `price`           int         NOT NULL COMMENT '所需兑换金额，单位：分',
    `stock`           int         NOT NULL COMMENT '积分商城商品库存',
    `activity_status` tinyint     NOT NULL COMMENT '积分商城商品状态: 0-开启, 1-禁用',
    `creator`         varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`     datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '积分商城商品表';

-- ----------------------------
-- Table structure for promotion_reward_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_reward_activity`;
CREATE TABLE `promotion_reward_activity`
(
    `id`                   bigint       NOT NULL AUTO_INCREMENT COMMENT '活动编号',
    `name`                 varchar(255) NOT NULL COMMENT '活动标题',
    `status`               tinyint      NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `start_time`           datetime     NOT NULL COMMENT '开始时间',
    `end_time`             datetime     NOT NULL COMMENT '结束时间',
    `remark`               varchar(512) NOT NULL COMMENT '备注',
    `condition_type`       tinyint      NOT NULL COMMENT '条件类型: 10-满N元, 20-满N件',
    `product_scope`        tinyint      NOT NULL COMMENT '商品范围: 1-全部商品, 2-指定商品, 3-指定品类',
    `product_scope_values` text         NOT NULL COMMENT '商品SPU编号列表',
    `rules`                text         NOT NULL COMMENT '优惠规则列表',
    `creator`              varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`              varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`          datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`              bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '满减送活动表';


-- ----------------------------
-- Table structure for promotion_seckill_activity
-- ----------------------------
DROP TABLE IF EXISTS `promotion_seckill_activity`;
CREATE TABLE `promotion_seckill_activity`
(
    `id`                 bigint       NOT NULL AUTO_INCREMENT COMMENT '秒杀活动编号',
    `spu_id`             bigint       NOT NULL COMMENT '秒杀活动商品SPU编号',
    `name`               varchar(255) NOT NULL COMMENT '秒杀活动名称',
    `status`             tinyint      NOT NULL COMMENT '活动状态: 0-开启, 1-禁用',
    `remark`             varchar(512) NOT NULL COMMENT '备注',
    `start_time`         datetime     NOT NULL COMMENT '活动开始时间',
    `end_time`           datetime     NOT NULL COMMENT '活动结束时间',
    `sort`               int          NOT NULL COMMENT '排序',
    `config_ids`         text         NOT NULL COMMENT '秒杀时段ID列表',
    `total_limit_count`  int          NOT NULL COMMENT '总限购数量',
    `single_limit_count` int          NOT NULL COMMENT '单次限购数量',
    `stock`              int          NOT NULL COMMENT '秒杀库存(剩余库存秒杀时扣减)',
    `total_stock`        int          NOT NULL COMMENT '秒杀总库存',
    `creator`            varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`            varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`            bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '秒杀活动表';

-- ----------------------------
-- Table structure for promotion_seckill_config
-- ----------------------------
DROP TABLE IF EXISTS `promotion_seckill_config`;
CREATE TABLE `promotion_seckill_config`
(
    `id`              bigint       NOT NULL AUTO_INCREMENT COMMENT '秒杀时段编号',
    `name`            varchar(255) NOT NULL COMMENT '秒杀时段名称',
    `start_time`      varchar(32)  NOT NULL COMMENT '开始时间点',
    `end_time`        varchar(32)  NOT NULL COMMENT '结束时间点',
    `slider_pic_urls` text         NOT NULL COMMENT '秒杀轮播图列表',
    `status`          tinyint      NOT NULL COMMENT '状态: 0-开启, 1-禁用',
    `creator`         varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '秒杀时段表';

-- ----------------------------
-- Table structure for promotion_seckill_product
-- ----------------------------
DROP TABLE IF EXISTS `promotion_seckill_product`;
CREATE TABLE `promotion_seckill_product`
(
    `id`                  bigint      NOT NULL AUTO_INCREMENT COMMENT '秒杀参与商品编号',
    `activity_id`         bigint      NOT NULL COMMENT '秒杀活动编号',
    `config_ids`          text        NOT NULL COMMENT '秒杀时段ID列表',
    `spu_id`              bigint      NOT NULL COMMENT '商品SPU编号',
    `sku_id`              bigint      NOT NULL COMMENT '商品SKU编号',
    `seckill_price`       int         NOT NULL COMMENT '秒杀金额，单位：分',
    `stock`               int         NOT NULL COMMENT '秒杀库存',
    `activity_status`     tinyint     NOT NULL COMMENT '秒杀商品状态: 0-开启, 1-禁用',
    `activity_start_time` datetime    NOT NULL COMMENT '活动开始时间点',
    `activity_end_time`   datetime    NOT NULL COMMENT '活动结束时间点',
    `creator`             varchar(64) NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`             varchar(64) NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`             bit(1)      NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '秒杀参与商品表';
