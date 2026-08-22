-- ============================================================
-- 表名: sys_pred_log
-- 模块: 日志模块
-- ============================================================
-- 设计思路:
-- 预测日志表，记录算法推理操作的完整日志。
-- 基于 algorithm_id + origin_md5 可实现缓存查询：相同算法处理相同图片直接返回历史结果。
-- status 字段（tinyint）追踪异步任务状态(1:处理中;2:已完成;3:失败;4:已取消)，配合 Python 后端异步处理。
-- 日志记录为只追加，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_pred_log`;
CREATE TABLE `sys_pred_log`
(
    `id`             bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `algorithm_id`   bigint                                                         NOT NULL COMMENT '算法id',
    `origin_file_id` bigint                                                         NULL DEFAULT NULL COMMENT '原始图像文件id（有雾图像）',
    `origin_md5`     char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL DEFAULT NULL COMMENT '原始图像md5值',
    `origin_url`     TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '原始图像url',
    `pred_file_id`   bigint                                                         NULL DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`       char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL DEFAULT NULL COMMENT '预测图像md5值',
    `pred_url`       TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '预测图像url',
    `time`           int                                                            NULL DEFAULT 0 COMMENT '推理时间（秒）',
    `status`         tinyint                                                        NOT NULL DEFAULT 2 COMMENT '任务状态(1:处理中;2:已完成;3:失败;4:已取消)',
    `error_message`  TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '失败错误信息',
    `create_time`    datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    INDEX `idx_origin_md5` (`origin_md5`) USING BTREE,
    INDEX `idx_pred_md5` (`pred_md5`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '模型预测日志表'
  ROW_FORMAT = DYNAMIC;
