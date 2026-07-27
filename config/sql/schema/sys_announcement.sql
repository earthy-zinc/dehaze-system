-- ============================================================
-- 表名: sys_announcement
-- 模块: 消息通知
-- ============================================================
-- 设计思路:
-- 系统公告表，存储管理员创建的公告定义（标题、内容、发送范围、定时发送等）。
-- 与 sys_message 的关系：公告发送时，系统根据 target_scope 批量生成 sys_message 记录（每接收人一条），
--   sys_message 的 biz_module='system'、biz_id=公告ID，实现公告内容与投递记录分离管理。
-- status 字段管理公告生命周期(1:草稿;2:待发送;3:已发送;4:已取消)，支持定时发送和取消。
-- target_scope + target_params 控制发送范围：all(全体)、level(按会员等级)、tag(按用户标签)、specified(指定用户)。
-- target_params(JSON) 存储范围参数，如 {"level": 2} 或 {"userIds": [1, 2, 3]}。
-- importance 字段区分普通/重要公告，重要公告在消息列表顶部置顶展示。
-- sent_count 记录实际送达人数，用于发送结果统计。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_announcement`;
CREATE TABLE `sys_announcement`
(
    `id`            bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `title`         varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '公告标题',
    `content`       TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NOT NULL COMMENT '公告内容',
    `type`          varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'operation' COMMENT '公告类型(maintenance:系统维护;feature:功能更新;activity:活动通知;operation:运营公告)',
    `importance`    tinyint                                                        NOT NULL DEFAULT 1 COMMENT '重要级别(1:普通;2:重要)',
    `target_scope`  varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'all' COMMENT '发送范围(all:全体用户;level:按会员等级;tag:按用户标签;specified:指定用户)',
    `target_params` json                                                           NULL DEFAULT NULL COMMENT '范围参数(JSON，如{"level":2}或{"userIds":[1,2,3]})',
    `status`        tinyint                                                        NOT NULL DEFAULT 1 COMMENT '公告状态(1:草稿;2:待发送;3:已发送;4:已取消)',
    `send_time`     datetime                                                       NULL DEFAULT NULL COMMENT '发送时间(定时发送时为计划时间)',
    `expire_time`   datetime                                                       NULL DEFAULT NULL COMMENT '过期时间(到期后公告从横幅移除)',
    `sent_count`    int                                                            NULL DEFAULT 0 COMMENT '已发送人数',
    `deleted`       tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`     bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`   datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_send_time` (`send_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '系统公告表'
  ROW_FORMAT = DYNAMIC;
