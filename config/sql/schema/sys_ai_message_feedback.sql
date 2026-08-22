-- ============================================================
-- 表名: sys_ai_message_feedback
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 消息反馈表，记录用户对 assistant 回复的点赞/点踩，用于质量分析和效果度量。
-- uk_message_user 唯一索引保证同一用户对同一消息只能反馈一次，再次反馈走 upsert 更新。
-- rating 取值 1:点赞/-1:点踩，comment 存储可选的反馈文字（点踩不强制填写，避免降低反馈率）。
-- tags 存储预设标签（JSON数组，如["accurate","detailed"]或["too_long","incorrect"]），比纯点赞点踩更有分析价值。
-- conversation_id/model/source 冗余存储，支撑按会话/模型/来源维度统计与归因，避免多表 JOIN。
-- processed/process_time 支撑反馈驱动闭环：点踩反馈由 XXL-Job 定时任务扫描 processed=0 的记录批量处理
--   （记忆提取/提示词优化/工具策略调整），处理成功置 1，失败下次扫描自动重试。
-- 反馈支持逻辑删除（用户撤销反馈），但撤销后再次反馈走 upsert 复活原行（类别①）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_message_feedback`;
CREATE TABLE `sys_ai_message_feedback`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `message_id`      bigint                                                         NOT NULL COMMENT '消息ID(仅assistant消息可反馈)',
    `user_id`         bigint                                                         NOT NULL COMMENT '用户ID',
    `conversation_id` bigint                                                         NULL DEFAULT NULL COMMENT '消息所属会话ID(冗余,支撑按会话维度统计)',
    `model`           varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '生成该消息的模型标识(按模型统计满意度与归因)',
    `source`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'internal' COMMENT '反馈来源(internal:内部API;compat:第三方兼容API)',
    `rating`          tinyint                                                        NOT NULL COMMENT '评分(1:点赞;-1:点踩)',
    `tags`            json                                                           NULL COMMENT '预设标签(JSON数组,点赞:accurate/detailed/concise/creative;点踩:incorrect/irrelevant/incomplete/too_long/bad_citation/harmful)',
    `comment`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '反馈内容(可选)',
    `processed`       tinyint                                                        NOT NULL DEFAULT 0 COMMENT '闭环处理状态(0:待处理;1:已处理,由XXL-Job定时扫描)',
    `process_time`    datetime                                                       NULL DEFAULT NULL COMMENT '闭环处理完成时间(支撑闭环时效统计)',
    `deleted`         tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_message_user` (`message_id`, `user_id`) USING BTREE,
    INDEX `idx_processed` (`processed`) USING BTREE,
    INDEX `idx_model` (`model`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI消息反馈表'
  ROW_FORMAT = DYNAMIC;
