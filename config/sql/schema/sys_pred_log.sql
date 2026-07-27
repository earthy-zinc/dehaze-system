-- ============================================================
-- 表名: sys_pred_log
-- 模块: 日志模块
-- ============================================================
-- 设计思路:
-- 预测日志表，记录算法推理操作的完整日志。
-- 基于 algorithm_id + origin_md5 可实现缓存查询：相同算法处理相同图片直接返回历史结果。
-- status 字段追踪异步任务状态（processing/completed/failed），配合 Python 后端异步处理。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_pred_log`;
CREATE TABLE `sys_pred_log`
(
    `id`             bigint   NOT NULL AUTO_INCREMENT COMMENT 'id',
    `algorithm_id`   bigint   NOT NULL COMMENT '算法id',
    `origin_file_id` bigint            DEFAULT NULL COMMENT '原始图像文件id（有雾图像）',
    `origin_md5`     char(32) NULL DEFAULT NULL COMMENT '原始图像md5值',
    `origin_url`     TEXT     NULL DEFAULT NULL COMMENT '原始图像url',
    `pred_file_id`   bigint            DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`       char(32) NULL DEFAULT NULL COMMENT '预测图像md5值',
    `pred_url`       TEXT     NULL DEFAULT NULL COMMENT '预测图像url',
    `time`           int               DEFAULT 0 COMMENT '推理时间（秒）',
    `status`         varchar(20) NOT NULL DEFAULT 'completed' COMMENT '任务状态：processing/completed/failed',
    `error_message`  text NULL DEFAULT NULL COMMENT '失败错误信息',
    `create_time`    datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint   NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint   NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    KEY `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    KEY `idx_origin_md5` (`origin_md5`) USING BTREE,
    KEY `idx_pred_md5` (`pred_md5`) USING BTREE,
    KEY `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='模型预测日志表';
