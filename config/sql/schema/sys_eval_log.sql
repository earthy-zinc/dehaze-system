-- ============================================================
-- 表名: sys_eval_log
-- 模块: 日志模块
-- ============================================================
-- 设计思路:
-- 评估日志表，记录算法评估操作日志。
-- result 字段使用 JSON 类型存储评估指标（PSNR/SSIM/LPIPS/NIQE/Entropy）。
-- 基于 algorithm_id + pred_md5 可实现评估缓存：相同算法对相同预测结果的评估不重复计算。
-- status 字段（tinyint）追踪异步任务状态(1:处理中;2:已完成;3:失败)。
-- 日志记录为只追加，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_eval_log`;
CREATE TABLE `sys_eval_log`
(
    `id`            bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `algorithm_id`  bigint                                                         NOT NULL COMMENT '算法id',
    `pred_file_id`  bigint                                                         NULL DEFAULT NULL COMMENT '预测图像文件id',
    `pred_md5`      char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL DEFAULT NULL COMMENT '预测图像md5值',
    `pred_url`      TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '预测图像url',
    `gt_file_id`    bigint                                                         NULL DEFAULT NULL COMMENT '真值图像文件id',
    `gt_md5`        char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL DEFAULT NULL COMMENT '真值图像md5值',
    `gt_url`        TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '真值图像url',
    `time`          int                                                            NULL DEFAULT 0 COMMENT '评估时间（秒）',
    `status`        tinyint                                                        NOT NULL DEFAULT 2 COMMENT '任务状态(1:处理中;2:已完成;3:失败)',
    `error_message` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '失败错误信息',
    `result`        json                                                           NULL COMMENT '评估指标结果（PSNR/SSIM/LPIPS/NIQE/Entropy等）',
    `create_time`   datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`     bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE,
    INDEX `idx_pred_md5` (`pred_md5`) USING BTREE,
    INDEX `idx_gt_md5` (`gt_md5`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '模型评估日志表'
  ROW_FORMAT = DYNAMIC;
