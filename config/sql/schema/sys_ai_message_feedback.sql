-- ============================================================
-- 表名: sys_ai_message_feedback
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 消息反馈表，记录用户对 assistant 回复的点赞/点踩，用于质量分析和效果度量。
-- uk_message_user 唯一索引保证同一用户对同一消息只能反馈一次，再次反馈走 upsert 更新。
-- rating 取值 1:点赞/-1:点踩，comment 存储可选的反馈文字。
-- tags 存储预设标签（JSON数组，如["accurate","detailed"]或["too_long","incorrect"]），比纯点赞点踩更有分析价值。
-- 反馈支持逻辑删除（用户撤销反馈），但撤销后再次反馈走 upsert 复活原行（类别①）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_message_feedback`;
CREATE TABLE `sys_ai_message_feedback`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `message_id`   bigint                                                         NOT NULL COMMENT '消息ID(仅assistant消息可反馈)',
    `user_id`      bigint                                                         NOT NULL COMMENT '用户ID',
    `rating`       tinyint                                                        NOT NULL COMMENT '评分(1:点赞;-1:点踩)',
    `tags`         json                                                           NULL COMMENT '预设标签(JSON数组,点赞:accurate/detailed/concise/creative;点踩:too_long/incorrect/irrelevant/harmful)',
    `comment`      TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '反馈内容(可选)',
    `deleted`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`  datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_message_user` (`message_id`, `user_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI消息反馈表'
  ROW_FORMAT = DYNAMIC;
