-- ============================================================
-- 表名: sys_ai_schedule
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 对话定时任务配置表，将对话中确认的处理流程固化为 Cron 定时任务（F-M08-009）。
-- user_id 归属用户，定时调度仅 VIP2 及以上用户可用；单用户任务上限 20 个。
-- cron 触发规则（5 位 Cron 表达式）；timezone 任务时区（默认 Asia/Shanghai，与配额重置时区一致）；
-- next_trigger_time 按任务时区计算，供列表排序与下次触发预览。
-- input(JSON) 输入来源：{type: fixed固定输入|dynamic动态拉取, ...}；output(JSON) 输出目标（消息推送/回调等）。
-- enabled 用户启停（1=启用，0=停用）；status 任务状态（1=正常，2=熔断停用：连续失败自动停用）。
-- circuit_streak 连续失败计数，达到阈值（默认 5 次，配置可调）时自动置 status=2 并通知用户；用户重新启用后清零。
-- 配置类表，使用逻辑删除（任务删除后不可恢复，无业务唯一键，标准软删即可）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_schedule`;
CREATE TABLE `sys_ai_schedule`
(
    `id`                bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`           bigint                                                          NOT NULL COMMENT '归属用户ID(关联sys_user.id)',
    `name`              varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '任务名称',
    `cron`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT 'Cron触发规则(5位Cron表达式,如"0 9 * * *")',
    `timezone`          varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL DEFAULT 'Asia/Shanghai' COMMENT '任务时区(触发时间计算时区,默认Asia/Shanghai)',
    `input`             json                                                            NULL COMMENT '输入来源JSON({type:fixed固定输入|dynamic动态拉取,...})',
    `output`            json                                                            NULL COMMENT '输出目标JSON(消息推送/回调等)',
    `enabled`           tinyint                                                         NOT NULL DEFAULT 1 COMMENT '用户启停(1:启用;0:停用)',
    `status`            tinyint                                                         NOT NULL DEFAULT 1 COMMENT '任务状态(1:正常;2:熔断停用,连续失败自动停用)',
    `circuit_streak`    int                                                             NOT NULL DEFAULT 0 COMMENT '连续失败计数(达到阈值自动熔断停用,重新启用后清零)',
    `next_trigger_time` datetime                                                        NULL DEFAULT NULL COMMENT '下次触发时间(按任务时区计算,供排序与预览)',
    `deleted`           tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`         bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`         bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`       datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`       datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user` (`user_id`) USING BTREE,
    INDEX `idx_next_trigger` (`next_trigger_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话定时任务配置表'
  ROW_FORMAT = DYNAMIC;
