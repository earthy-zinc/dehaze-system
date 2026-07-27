-- ============================================================
-- 表名: sys_message
-- 模块: 消息通知
-- ============================================================
-- 设计思路:
-- 消息主表，存储所有投递给用户的消息记录（站内信/业务通知/会员通知/告警等）。
-- 每条消息一行记录，recipient_id 标识接收人，type 区分消息类型。
-- 业务模块通过 biz_module + biz_id 实现幂等去重（如同一订单号不重复生成退款通知）。
-- read_status 字段管理已读/未读状态，用户阅读后自动标记并记录 read_time。
-- deleted 字段为用户侧软删除（仅从当前用户视图移除），系统按 expires_at 定时物理清理过期记录。
-- priority 字段控制推送策略：紧急消息走全渠道，低优先级仅站内信。
-- extra 字段使用 JSON 存储跳转参数等结构化扩展数据。
-- (recipient_id, read_status) 复合索引优化未读消息高频查询，(recipient_id, deleted, create_time) 复合索引优化列表分页。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_message`;
CREATE TABLE `sys_message`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `type`         varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '消息类型(inbox:站内信;announcement:系统公告;business:业务通知;member:会员通知;alert:告警通知;critical_alert:严重告警)',
    `title`        varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '消息标题',
    `content`      TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NOT NULL COMMENT '消息正文',
    `sender_type`  tinyint                                                        NOT NULL DEFAULT 1 COMMENT '发送者类型(1:系统;2:管理员)',
    `recipient_id` bigint                                                         NOT NULL COMMENT '接收人ID',
    `biz_module`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '业务模块(member:会员管理;order:订单管理;feedback:反馈评价;prediction:去雾处理;system:系统)',
    `biz_id`       varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '业务ID(订单号/反馈ID等，用于幂等去重和跳转)',
    `priority`     tinyint                                                        NOT NULL DEFAULT 2 COMMENT '优先级(1:低;2:中;3:高;4:紧急)',
    `jump_url`     varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '跳转链接(相对路径，如/member/profile)',
    `extra`        json                                                           NULL DEFAULT NULL COMMENT '扩展数据(JSON)',
    `read_status`  tinyint                                                        NOT NULL DEFAULT 0 COMMENT '已读状态(0:未读;1:已读)',
    `read_time`    datetime                                                       NULL DEFAULT NULL COMMENT '已读时间',
    `deleted`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '用户删除标识(0:未删除;1:已删除)',
    `expires_at`   datetime                                                       NULL DEFAULT NULL COMMENT '过期时间(到期后系统自动清理)',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID(系统消息为NULL)',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`  datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_recipient_read` (`recipient_id`, `read_status`) USING BTREE,
    INDEX `idx_recipient_list` (`recipient_id`, `deleted`, `create_time`) USING BTREE,
    INDEX `idx_biz_dedup` (`biz_module`, `biz_id`) USING BTREE,
    INDEX `idx_expires_at` (`expires_at`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '消息表'
  ROW_FORMAT = DYNAMIC;
