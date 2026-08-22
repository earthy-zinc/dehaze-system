-- ============================================================
-- 表名: sys_ai_schedule_run
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 对话定时任务执行历史表。每次触发/跳过/执行均写入一行，支撑执行观测与
-- "为什么没跑 / 重复跑 / 成本异常"排查。
-- 幂等防重入：uk_schedule_window 唯一约束(schedule_id, window_start)，服务重启、
-- 多实例并发扫描、时钟漂移均不产生重复执行（数据库唯一约束兜底，不依赖内存）。
-- window_start 触发窗口（按任务周期对齐，如每天任务为当日 0 点），同窗口不重复执行。
-- status 执行结果；skip_reason 记录跳过原因（overlap 任务重叠/quota 配额不足/
-- circuit 熔断停用/idempotent 幂等去重）。
-- credits/duration_ms 消耗积分与耗时；error_msg 失败原因；conversation_id 关联
-- 执行产生的对话；request_id 关联调用链路日志。
-- 日志类表，只追加不逻辑删除；保留 30 天由定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_schedule_run`;
CREATE TABLE `sys_ai_schedule_run`
(
    `id`              bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `schedule_id`     bigint                                                          NOT NULL COMMENT '关联定时任务ID(关联sys_ai_schedule.id)',
    `user_id`         bigint                                                          NOT NULL COMMENT '归属用户ID(关联sys_user.id,幂等键组成部分)',
    `window_start`    datetime                                                        NOT NULL COMMENT '触发窗口(幂等键组成部分,按任务周期对齐)',
    `status`          tinyint                                                         NOT NULL DEFAULT 1 COMMENT '执行结果(1:成功;2:失败;3:跳过)',
    `skip_reason`     varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL COMMENT '跳过原因(overlap:任务重叠;quota:配额不足;circuit:熔断停用;idempotent:幂等去重)',
    `credits`         decimal(10, 4)                                                  NULL DEFAULT NULL COMMENT '本次执行消耗积分',
    `duration_ms`     int                                                             NULL DEFAULT NULL COMMENT '执行耗时(毫秒)',
    `error_msg`       varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL COMMENT '失败原因',
    `conversation_id` bigint                                                          NULL DEFAULT NULL COMMENT '执行产生的会话ID(关联sys_ai_conversation.id)',
    `request_id`      varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL COMMENT '调用链路ID(关联日志排查)',
    `create_time`     datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_schedule_window` (`schedule_id`, `window_start`) USING BTREE,
    INDEX `idx_user` (`user_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话定时任务执行历史表'
  ROW_FORMAT = DYNAMIC;
