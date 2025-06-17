CREATE DATABASE IF NOT EXISTS `pei_report`;
USE `pei_report`;


-- ----------------------------
-- Table structure for report_go_view_project
-- ----------------------------
DROP TABLE IF EXISTS `report_go_view_project`;
CREATE TABLE `report_go_view_project`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '编号，数据库自增',
    `name`        varchar(255)  NOT NULL COMMENT '项目名称',
    `pic_url`     varchar(1024) NOT NULL COMMENT '预览图片 URL',
    `content`     text          NOT NULL COMMENT '报表内容 JSON 配置，使用字符串存储',
    `status`      tinyint       NOT NULL COMMENT '发布状态 (0 - 已发布, 1 - 未发布)',
    `remark`      text COMMENT '项目备注',
    `create_time` datetime      NOT NULL COMMENT '创建时间',
    `update_time` datetime      NOT NULL COMMENT '最后更新时间',
    `creator`     varchar(64)   NOT NULL COMMENT '创建者',
    `updater`     varchar(64)   NOT NULL COMMENT '更新者',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'GoView 项目表';
