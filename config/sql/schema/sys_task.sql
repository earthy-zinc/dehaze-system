-- ============================================================
-- 表名: sys_task
-- 模块: 日志模块
-- ============================================================
-- 设计思路:
-- 异步任务调度主表，配合 RabbitMQ 实现生产消费分离。
-- task_id (UUID) 唯一索引，作为业务层任务标识。
-- idempotency_key 唯一索引实现 HTTP 幂等：客户端通过 Idempotency-Key 头防止重复提交。
-- 所有导入导出任务复用此表，通过 task_type 区分模块和操作类型。
-- retry_count 记录 MQ 重试次数，worker_id 标识执行 Worker，支持任务追踪和负载分析。
-- status 字段（tinyint）使用 5 状态机：1:待处理/2:处理中/3:已完成/4:失败/5:已取消。
-- params 与 result 字段使用原生 json 类型存储结构化数据。
-- 任务记录为只追加，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_task`;
CREATE TABLE `sys_task`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `task_id`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '任务ID（UUID）',
    `task_type`       varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '任务类型',
    `status`          tinyint                                                        NOT NULL DEFAULT 1 COMMENT '任务状态(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)',
    `progress`        int                                                            NULL DEFAULT 0 COMMENT '任务进度（百分比）',
    `total_files`     int                                                            NULL DEFAULT 0 COMMENT '总文件数',
    `processed_files` int                                                            NULL DEFAULT 0 COMMENT '已处理文件数',
    `params`          json                                                           NULL COMMENT '任务参数（JSON）',
    `result`          json                                                           NULL COMMENT '任务结果（下载链接等）',
    `error_message`   TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '错误信息',
    `started_at`      datetime                                                       NULL DEFAULT NULL COMMENT '开始时间',
    `completed_at`    datetime                                                       NULL DEFAULT NULL COMMENT '完成时间',
    `expires_at`      datetime                                                       NULL DEFAULT NULL COMMENT '过期时间',
    `idempotency_key` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '客户端幂等键（HTTP Idempotency-Key 头）',
    `retry_count`     int                                                            NOT NULL DEFAULT 0 COMMENT 'MQ 重试次数',
    `worker_id`       varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '执行 Worker 标识',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_task_id` (`task_id`) USING BTREE,
    UNIQUE INDEX `uk_idempotency_key` (`idempotency_key`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_create_by` (`create_by`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '系统任务表'
  ROW_FORMAT = DYNAMIC;
