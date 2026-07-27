-- ============================================================
-- 表名: sys_feedback
-- 模块: 商业化模块-反馈评价
-- ============================================================
-- 设计思路:
-- 用户反馈表，收集对产品功能的意见反馈。
-- feedback_type 标识四种类型（功能建议/问题报告/体验反馈/投诉）。
-- status 标识处理状态机（待处理/处理中/已回复/已关闭），assignee_id 记录处理人。
-- priority 标识优先级（普通/紧急/高优），支撑后台按优先级排序。
-- images 使用 JSON 数组存储截图URL（最多5张）。
-- tags 使用 JSON 数组存储后台打的标签（高频问题/已知问题/已排期等）。
-- 关闭原因记录在 close_reason，便于统计分析。
-- contact 联系方式仅管理员可见，前端脱敏。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_feedback`;
CREATE TABLE `sys_feedback`
(
    `id`             bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`        bigint                                                         NOT NULL COMMENT '提交用户ID',
    `feedback_type`  varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '反馈类型(suggestion:功能建议;bug:问题报告;experience:体验反馈;complaint:投诉)',
    `title`          varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '反馈标题',
    `content`        varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '反馈内容',
    `contact`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '联系方式（手机/邮箱，仅管理员可见）',
    `images`         json                                                           NULL DEFAULT NULL COMMENT '截图URL（JSON数组，最多5张）',
    `related_module` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '相关模块',
    `status`         tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:待处理;2:处理中;3:已回复;4:已关闭)',
    `priority`       tinyint                                                        NOT NULL DEFAULT 1 COMMENT '优先级(1:普通;2:紧急;3:高优)',
    `assignee_id`    bigint                                                         NULL DEFAULT NULL COMMENT '处理人ID',
    `assigned_time`  datetime                                                       NULL DEFAULT NULL COMMENT '分配时间',
    `tags`           json                                                           NULL DEFAULT NULL COMMENT '反馈标签（JSON数组，后台打标）',
    `close_reason`   varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '关闭原因',
    `deleted`        tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_feedback_type` (`feedback_type`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_priority_status` (`priority`, `status`) USING BTREE,
    INDEX `idx_assignee_id` (`assignee_id`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户反馈表'
  ROW_FORMAT = DYNAMIC;
