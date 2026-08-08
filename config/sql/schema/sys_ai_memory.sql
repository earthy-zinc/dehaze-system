-- ============================================================
-- 表名: sys_ai_memory
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 长期记忆表，存储跨会话持久化的记忆，参考认知心理学记忆分类模型
--   （Endel Tulving：情景记忆/语义记忆/程序记忆）。
-- 此表为记忆的源头数据，写入后同步到 ES 做向量检索，对话时按需注入 LLM prompt。
- memory_type 分类记忆：
  · episodic（情景记忆）：带时空标签的事件记录，"某时某地发生了什么"
  · semantic（语义记忆）：抽象知识/事实/用户偏好，"知道什么"
  · procedural（程序记忆）：操作技能/流程/工具使用方法，"会做什么"
- importance 为记忆重要性评分（0-100），综合情感/频率/时效/信息增益/显式标记计算，
  支撑记忆整理的优先级排序和遗忘策略。
- last_accessed_at 为最后访问时间，配合 create_time/update_time 支撑遗忘曲线衰减计算：
  priority = importance × exp(-Δt / half_life)
- access_count 为检索命中次数，被检索的记忆"重激活"（重置衰减计时器），类似人类"复习巩固"。
- source 标识记忆来源（conversation:对话提取;feedback:反馈提取;reflection:反思整合;manual:手动录入）。
- status 控制记忆启停，禁用的记忆不注入 LLM 但保留记录；archived 标记被遗忘策略归档的记忆。
- 记忆支持逻辑删除（用户可清除某条记忆），无业务唯一键，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_memory`;
CREATE TABLE `sys_ai_memory`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`         bigint                                                         NOT NULL COMMENT '用户ID',
    `memory_type`     varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '记忆类型(episodic:情景记忆;semantic:语义记忆;procedural:程序记忆)',
    `content`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NOT NULL COMMENT '记忆内容',
    `metadata`        json                                                           NULL COMMENT '结构化属性(情景记忆:时间戳/事件/结果;语义记忆:偏好/事实;程序记忆:技能/流程)',
    `importance`      tinyint                                                        NOT NULL DEFAULT 50 COMMENT '重要性评分(0-100,综合情感/频率/时效/信息增益/显式标记计算)',
    `access_count`    int                                                            NOT NULL DEFAULT 0 COMMENT '检索命中次数(被检索的记忆重激活,重置衰减计时器)',
    `last_accessed_at` datetime                                                       NULL DEFAULT NULL COMMENT '最后访问时间(支撑遗忘曲线衰减计算)',
    `source`          varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'conversation' COMMENT '来源(conversation:对话提取;feedback:反馈提取;reflection:反思整合;manual:手动录入)',
    `status`          tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `archived`        tinyint                                                        NOT NULL DEFAULT 0 COMMENT '是否被遗忘策略归档(0:未归档;1:已归档,不再注入但保留记录)',
    `deleted`         tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_type` (`user_id`, `memory_type`) USING BTREE,
    INDEX `idx_user_active` (`user_id`, `status`, `archived`, `deleted`) USING BTREE,
    INDEX `idx_importance` (`importance` DESC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI长期记忆表'
  ROW_FORMAT = DYNAMIC;
