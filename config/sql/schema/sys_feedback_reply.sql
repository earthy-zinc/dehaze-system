-- ============================================================
-- 表名: sys_feedback_reply
-- 模块: 商业化模块-反馈评价
-- ============================================================
-- 设计思路:
-- 反馈回复表，记录反馈处理过程中的多次回复。
-- replier_type 区分用户补充（1）和管理员回复（2），支撑处理时间线展示。
-- reply_type 标识回复类型（信息补充/已解决/暂不支持/转开发）。
-- attachments 使用 JSON 数组存储附件URL。
-- 回复记录为只追加，不使用逻辑删除。
-- 复合索引 (feedback_id, create_time) 优化按反馈查看回复列表。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_feedback_reply`;
CREATE TABLE `sys_feedback_reply`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `feedback_id`  bigint                                                         NOT NULL COMMENT '反馈ID',
    `replier_id`   bigint                                                         NOT NULL COMMENT '回复人ID',
    `replier_type` tinyint                                                        NOT NULL COMMENT '回复人类型(1:用户;2:管理员)',
    `content`      varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '回复内容',
    `reply_type`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '回复类型(info:信息补充;resolved:已解决;unsupported:暂不支持;dev_transfer:转开发)',
    `attachments`  json                                                           NULL DEFAULT NULL COMMENT '附件URL（JSON数组）',
    `create_time`  datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_feedback_id_create_time` (`feedback_id`, `create_time`) USING BTREE,
    INDEX `idx_replier_id` (`replier_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '反馈回复表'
  ROW_FORMAT = DYNAMIC;
