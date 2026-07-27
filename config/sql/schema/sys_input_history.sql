-- ============================================================
-- 表名: sys_input_history
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 图像输入历史表，记录用户每次去雾处理的完整信息。
-- algorithm_name 冗余存储算法名称，避免 join 查询。
-- idx_user_time 复合索引优化「用户历史按时间倒序」的高频查询。
-- idx_user_favorite 索引优化「用户收藏列表」查询。
-- sync_status 标识是否已同步到远端，用于多端数据同步场景。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_input_history`;
CREATE TABLE `sys_input_history`
(
    `id`                    bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`               bigint       NOT NULL COMMENT '用户ID',
    `original_image_url`    varchar(500) NULL COMMENT '原始图片URL',
    `original_thumbnail_url` varchar(500) NULL COMMENT '原始缩略图URL',
    `result_image_url`      varchar(500) NULL COMMENT '处理结果图片URL',
    `result_thumbnail_url`  varchar(500) NULL COMMENT '结果缩略图URL',
    `algorithm_id`          bigint       NULL COMMENT '算法ID',
    `algorithm_name`        varchar(100) NULL COMMENT '算法名称（冗余）',
    `algorithm_params`      TEXT         NULL COMMENT '算法参数（JSON）',
    `processing_time`       int          NULL COMMENT '处理耗时（毫秒）',
    `status`                tinyint      NULL DEFAULT 3 COMMENT '处理状态（1=成功，2=失败，3=处理中）',
    `input_source`          varchar(20)  NULL COMMENT '图片来源（upload/camera/sample）',
    `is_favorite`           tinyint(1)   NULL DEFAULT 0 COMMENT '是否收藏',
    `sync_status`           tinyint      NULL DEFAULT 0 COMMENT '同步状态（0=未同步，1=已同步）',
    `create_time`           datetime     NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`           datetime     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`             bigint       NULL COMMENT '创建人ID',
    `update_by`             bigint       NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_time` (`user_id`, `create_time` DESC) USING BTREE,
    INDEX `idx_user_favorite` (`user_id`, `is_favorite`, `create_time` DESC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='图像输入历史记录表';
