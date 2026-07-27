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
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_task`;
CREATE TABLE `sys_task`
(
    `id`              BIGINT      NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `task_id`         VARCHAR(64) NOT NULL COMMENT '任务ID（UUID）',
    `task_type`       VARCHAR(32) NOT NULL COMMENT '任务类型',
    `status`          VARCHAR(32) NOT NULL DEFAULT 'PENDING' COMMENT '任务状态',
    `progress`        INT                  DEFAULT 0 COMMENT '任务进度（百分比）',
    `total_files`     INT                  DEFAULT 0 COMMENT '总文件数',
    `processed_files` INT                  DEFAULT 0 COMMENT '已处理文件数',
    `params`          TEXT COMMENT '任务参数（JSON）',
    `result`          TEXT COMMENT '任务结果（下载链接）',
    `error_message`   TEXT COMMENT '错误信息',
    `create_by`       BIGINT COMMENT '创建人ID',
    `update_by`       BIGINT COMMENT '修改人ID',
    `create_time`     DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     DATETIME    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `started_at`      DATETIME COMMENT '开始时间',
    `completed_at`    DATETIME COMMENT '完成时间',
    `expires_at`      DATETIME COMMENT '过期时间',
    `idempotency_key` VARCHAR(64) COMMENT '客户端幂等键（HTTP Idempotency-Key 头）',
    `retry_count`     INT         NOT NULL DEFAULT 0 COMMENT 'MQ 重试次数',
    `worker_id`       VARCHAR(64) COMMENT '执行 Worker 标识',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `idx_task_id` (`task_id`) USING BTREE,
    UNIQUE INDEX `idx_idempotency_key` (`idempotency_key`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_create_by` (`create_by`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='系统任务表';
