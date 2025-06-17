CREATE DATABASE IF NOT EXISTS `pei_mp`;
USE `pei_mp`;

-- ----------------------------
-- Table structure for mp_account
-- ----------------------------
DROP TABLE IF EXISTS `mp_account`;
CREATE TABLE `mp_account`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '公众号账号编号',
    `name`        varchar(255)  NOT NULL COMMENT '公众号名称',
    `account`     varchar(64)   NOT NULL COMMENT '公众号账号',
    `app_id`      varchar(64)   NOT NULL COMMENT '公众号appid',
    `app_secret`  varchar(128)  NOT NULL COMMENT '公众号密钥',
    `token`       varchar(128)  NOT NULL COMMENT '公众号token',
    `aes_key`     varchar(128)  NOT NULL COMMENT '消息加解密密钥',
    `qr_code_url` varchar(1024) NOT NULL COMMENT '二维码图片URL',
    `remark`      varchar(512)  NOT NULL COMMENT '备注',
    `creator`     varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`   bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号账号表';

-- ----------------------------
-- Table structure for mp_material
-- ----------------------------
DROP TABLE IF EXISTS `mp_material`;
CREATE TABLE `mp_material`
(
    `id`           bigint        NOT NULL AUTO_INCREMENT COMMENT '素材主键',
    `account_id`   bigint        NOT NULL COMMENT '公众号账号编号',
    `app_id`       varchar(64)   NOT NULL COMMENT '公众号appid',
    `media_id`     varchar(128)  NOT NULL COMMENT '公众号素材id',
    `type`         varchar(32)   NOT NULL COMMENT '文件类型',
    `permanent`    bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否永久素材: 0-否, 1-是',
    `url`          varchar(1024) NOT NULL COMMENT '文件服务器URL',
    `name`         varchar(255)  NOT NULL COMMENT '名字',
    `mp_url`       varchar(1024) NOT NULL COMMENT '公众号文件URL',
    `title`        varchar(255)  NOT NULL COMMENT '视频素材标题',
    `introduction` varchar(512)  NOT NULL COMMENT '视频素材描述',
    `creator`      varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号素材表';
-- ----------------------------
-- Table structure for mp_menu
-- ----------------------------
DROP TABLE IF EXISTS `mp_menu`;
CREATE TABLE `mp_menu`
(
    `id`                     bigint        NOT NULL AUTO_INCREMENT COMMENT '菜单编号',
    `account_id`             bigint        NOT NULL COMMENT '公众号账号编号',
    `app_id`                 varchar(64)   NOT NULL COMMENT '公众号appid',
    `name`                   varchar(64)   NOT NULL COMMENT '菜单名称',
    `menu_key`               varchar(255)  NOT NULL COMMENT '菜单标识',
    `parent_id`              bigint        NOT NULL DEFAULT 0 COMMENT '父菜单编号: 0-顶级菜单',
    `type`                   varchar(64)   NOT NULL COMMENT '按钮类型',
    `url`                    varchar(1024) NOT NULL COMMENT '网页链接',
    `mini_program_app_id`    varchar(64)   NOT NULL COMMENT '小程序AppId',
    `mini_program_page_path` varchar(255)  NOT NULL COMMENT '小程序页面路径',
    `article_id`             varchar(128)  NOT NULL COMMENT '跳转图文媒体编号',
    `reply_message_type`     varchar(32)   NOT NULL COMMENT '回复消息类型',
    `reply_content`          text          NOT NULL COMMENT '回复消息内容',
    `reply_media_id`         varchar(128)  NOT NULL COMMENT '回复媒体id',
    `reply_media_url`        varchar(1024) NOT NULL COMMENT '回复媒体URL',
    `reply_title`            varchar(255)  NOT NULL COMMENT '回复标题',
    `reply_description`      varchar(512)  NOT NULL COMMENT '回复描述',
    `reply_thumb_media_id`   varchar(128)  NOT NULL COMMENT '回复缩略图媒体id',
    `reply_thumb_media_url`  varchar(1024) NOT NULL COMMENT '回复缩略图媒体URL',
    `reply_articles`         text          NULL COMMENT '回复图文消息数组',
    `reply_music_url`        varchar(1024) NOT NULL COMMENT '回复音乐链接',
    `reply_hq_music_url`     varchar(1024) NOT NULL COMMENT '回复高质量音乐链接',
    `creator`                varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`            datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`            datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号菜单表';
-- ----------------------------
-- Table structure for mp_auto_reply
-- ----------------------------
DROP TABLE IF EXISTS `mp_auto_reply`;
CREATE TABLE `mp_auto_reply`
(
    `id`                       bigint        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `account_id`               bigint        NOT NULL COMMENT '公众号账号编号',
    `app_id`                   varchar(64)   NOT NULL COMMENT '公众号appid',
    `type`                     tinyint       NOT NULL COMMENT '回复类型: 1-关注回复, 2-消息回复, 3-关键字回复',
    `request_keyword`          varchar(255)  NOT NULL COMMENT '请求关键字',
    `request_match`            tinyint       NOT NULL COMMENT '请求关键字匹配方式: 1-完全匹配, 2-包含匹配',
    `request_message_type`     varchar(32)   NOT NULL COMMENT '请求消息类型',
    `response_message_type`    varchar(32)   NOT NULL COMMENT '回复消息类型',
    `response_content`         text          NOT NULL COMMENT '回复消息内容',
    `response_media_id`        varchar(128)  NOT NULL COMMENT '回复媒体id',
    `response_media_url`       varchar(1024) NOT NULL COMMENT '回复媒体URL',
    `response_title`           varchar(255)  NOT NULL COMMENT '回复标题',
    `response_description`     varchar(512)  NOT NULL COMMENT '回复描述',
    `response_thumb_media_id`  varchar(128)  NOT NULL COMMENT '回复缩略图媒体id',
    `response_thumb_media_url` varchar(1024) NOT NULL COMMENT '回复缩略图媒体URL',
    `response_articles`        text          NULL COMMENT '回复图文消息数组',
    `response_music_url`       varchar(1024) NOT NULL COMMENT '回复音乐链接',
    `response_hq_music_url`    varchar(1024) NOT NULL COMMENT '回复高质量音乐链接',
    `creator`                  varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`              datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`                  varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`              datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`                  bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号自动回复规则表';

-- ----------------------------
-- Table structure for mp_message
-- ----------------------------
DROP TABLE IF EXISTS `mp_message`;
CREATE TABLE `mp_message`
(
    `id`              bigint         NOT NULL AUTO_INCREMENT COMMENT '消息主键',
    `msg_id`          bigint         NOT NULL COMMENT '微信公众号消息id',
    `account_id`      bigint         NOT NULL COMMENT '公众号账号编号',
    `app_id`          varchar(64)    NOT NULL COMMENT '公众号appid',
    `user_id`         bigint         NOT NULL COMMENT '公众号粉丝编号',
    `openid`          varchar(64)    NOT NULL COMMENT '公众号粉丝标志',
    `type`            varchar(32)    NOT NULL COMMENT '消息类型',
    `send_from`       tinyint        NOT NULL COMMENT '消息来源: 0-用户发送, 1-系统回复',
    `content`         text           NOT NULL COMMENT '消息内容',
    `media_id`        varchar(128)   NOT NULL COMMENT '媒体文件编号',
    `media_url`       varchar(1024)  NOT NULL COMMENT '媒体文件URL',
    `recognition`     varchar(255)   NOT NULL COMMENT '语音识别后文本',
    `format`          varchar(32)    NOT NULL COMMENT '语音格式',
    `title`           varchar(255)   NOT NULL COMMENT '标题',
    `description`     varchar(512)   NOT NULL COMMENT '描述',
    `thumb_media_id`  varchar(128)   NOT NULL COMMENT '缩略图媒体id',
    `thumb_media_url` varchar(1024)  NOT NULL COMMENT '缩略图媒体URL',
    `url`             varchar(1024)  NOT NULL COMMENT '图文消息跳转链接',
    `location_x`      decimal(10, 6) NOT NULL COMMENT '地理位置维度',
    `location_y`      decimal(10, 6) NOT NULL COMMENT '地理位置经度',
    `scale`           decimal(10, 2) NOT NULL COMMENT '地图缩放大小',
    `label`           varchar(255)   NOT NULL COMMENT '详细地址',
    `articles`        text           NULL COMMENT '图文消息数组',
    `music_url`       varchar(1024)  NOT NULL COMMENT '音乐链接',
    `hq_music_url`    varchar(1024)  NOT NULL COMMENT '高质量音乐链接',
    `event`           varchar(64)    NOT NULL COMMENT '事件类型',
    `event_key`       varchar(255)   NOT NULL COMMENT '事件Key',
    `creator`         varchar(64)    NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)    NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`     datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号消息表';
-- ----------------------------
-- Table structure for mp_tag
-- ----------------------------
DROP TABLE IF EXISTS `mp_tag`;
CREATE TABLE `mp_tag`
(
    `id`          bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `tag_id`      bigint       NOT NULL COMMENT '公众号标签id',
    `name`        varchar(255) NOT NULL COMMENT '标签名',
    `count`       int          NOT NULL COMMENT '此标签下粉丝数',
    `account_id`  bigint       NOT NULL COMMENT '公众号账号编号',
    `app_id`      varchar(64)  NOT NULL COMMENT '公众号appid',
    `creator`     varchar(64)  NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)  NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time` datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '公众号标签表';
-- ----------------------------
-- Table structure for mp_user
-- ----------------------------
DROP TABLE IF EXISTS `mp_user`;
CREATE TABLE `mp_user`
(
    `id`               bigint        NOT NULL AUTO_INCREMENT COMMENT '粉丝编号',
    `openid`           varchar(64)   NOT NULL COMMENT '粉丝标识',
    `union_id`         varchar(64)   NOT NULL COMMENT '微信生态唯一标识',
    `subscribe_status` tinyint       NOT NULL COMMENT '关注状态: 0-已关注, 1-取消关注',
    `subscribe_time`   datetime      NOT NULL COMMENT '关注时间',
    `unsubscribe_time` datetime      NOT NULL COMMENT '取消关注时间',
    `nickname`         varchar(64)   NOT NULL COMMENT '昵称',
    `head_image_url`   varchar(1024) NOT NULL COMMENT '头像地址',
    `language`         varchar(32)   NOT NULL COMMENT '语言',
    `country`          varchar(64)   NOT NULL COMMENT '国家',
    `province`         varchar(64)   NOT NULL COMMENT '省份',
    `city`             varchar(64)   NOT NULL COMMENT '城市',
    `remark`           varchar(255)  NOT NULL COMMENT '备注',
    `tag_ids`          text          NOT NULL COMMENT '标签编号列表',
    `account_id`       bigint        NOT NULL COMMENT '公众号账号编号',
    `app_id`           varchar(64)   NOT NULL COMMENT '公众号appid',
    `creator`          varchar(64)   NOT NULL DEFAULT '' COMMENT '创建者',
    `create_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)   NOT NULL DEFAULT '' COMMENT '更新者',
    `update_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = '微信公众号粉丝表';
