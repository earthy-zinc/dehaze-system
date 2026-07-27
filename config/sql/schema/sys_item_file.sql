-- ============================================================
-- 表名: sys_item_file
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 数据项-文件多对多关联表，承载图片类型(hazy/clear/trans/depth/segment)和场景元数据。
-- haze_level 支持多种标注规范：人工分级(light/medium/heavy)、β参数、A+β双参数。
-- thumbnail_file_id 冗余存储缩略图文件 ID，避免二次查询 sys_file。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_item_file`;
CREATE TABLE `sys_item_file`
(
    `id`                bigint      NOT NULL AUTO_INCREMENT COMMENT 'id',
    `item_id`           bigint      NOT NULL COMMENT '所属数据项id',
    `file_id`           bigint      NOT NULL COMMENT '文件id',
    `thumbnail_file_id` bigint       DEFAULT NULL COMMENT '缩略图文件id',
    `type`              varchar(64) NOT NULL COMMENT '图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)',
    `description`       varchar(255) DEFAULT NULL COMMENT '描述',
    `scene_type`        varchar(64)  DEFAULT NULL COMMENT '场景类型',
    `haze_level`        varchar(32)  DEFAULT NULL COMMENT '雾霾程度标识，支持多种规范：light/medium/heavy（人工分级），beta=0.5（β参数），A=0.8,beta=0.2（大气光A+β双参数），空值表示未标注或无雾',
    `width`             int          DEFAULT NULL COMMENT '图片宽度',
    `height`            int          DEFAULT NULL COMMENT '图片高度',
    `usage_count`       bigint       DEFAULT 0 COMMENT '使用次数',
    `create_time`       datetime     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_item_id_file_id` (`item_id`, `file_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='数据项图片关联表';
