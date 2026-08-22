-- ============================================================
-- 表名: sys_ai_message
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 对话消息表，记录会话中的每一条消息（system/user/assistant/tool）。
-- role 枚举为 system/user/assistant/tool，覆盖 OpenAI/Claude 消息角色体系。
-- parent_message_id 支持分支对话（重新生成、编辑后重发），建立消息树状关系。
-- content 使用 LONGTEXT 存储消息正文（助手回复可能很长）。
-- tool_calls(JSON) 存储 assistant 消息触发的工具调用列表（一次可触发多个工具），
-- tool_call_id 用于 role=tool 的消息关联对应的工具调用（与 OpenAI tool_call 机制对齐）。
-- model 记录本条消息实际使用的模型（一次会话可能切换模型）。
-- status 管理消息生命周期：1:流式输出中/2:已完成/3:失败/4:已取消，流式断线重连依赖此字段。
-- error 记录失败信息；metadata(JSON) 存储多模态读取次数、工具调用耗时等灵活元数据。
-- input_tokens/output_tokens/cached_input_tokens 记录 token 消耗，credits 记录换算后积分。
-- task_id 关联 sys_task，用于异步等待场景（如图像处理完成后恢复推理）。
-- edited 标识消息是否被编辑过（用户编辑重发后，原消息标记 edited=1 并保留 original_content）。
-- original_content 存储编辑前的原文（edited=1 时填充，支撑前端展示"已编辑"标识和查看编辑历史）。
-- 消息支持逻辑删除，无业务唯一键，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_message`;
CREATE TABLE `sys_ai_message`
(
    `id`                   bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `conversation_id`      bigint                                                          NOT NULL COMMENT '会话ID(关联sys_ai_conversation.id)',
    `parent_message_id`    bigint                                                          NULL DEFAULT NULL COMMENT '父消息ID(支持分支对话:重新生成/编辑后重发)',
    `role`                 varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '消息角色(system;user;assistant;tool)',
    `content`              LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL COMMENT '消息内容',
    `tool_calls`           json                                                            NULL COMMENT '工具调用列表(assistant消息触发，含tool_name/arguments/tool_call_id)',
    `tool_call_id`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '工具调用结果关联ID(role=tool时关联对应的tool_call)',
    `model`                varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '本条消息使用的模型标识(一次会话可能切换模型)',
    `status`               tinyint                                                         NOT NULL DEFAULT 1 COMMENT '消息状态(1:流式输出中;2:已完成;3:失败;4:已取消)',
    `error`                TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '错误信息(status=3时填充)',
    `metadata`             json                                                            NULL COMMENT '元数据(多模态读取次数;工具调用耗时;RAG检索命中等)',
    `input_tokens`         int                                                             NOT NULL DEFAULT 0 COMMENT '输入Token数(含缓存命中部分)',
    `output_tokens`        int                                                             NOT NULL DEFAULT 0 COMMENT '输出Token数',
    `cached_input_tokens`  int                                                             NOT NULL DEFAULT 0 COMMENT '其中缓存命中的输入Token数',
    `credits`              bigint                                                          NOT NULL DEFAULT 0 COMMENT '消耗积分数(按模型计费比例换算后)',
    `task_id`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '关联异步任务ID(等待图像处理等异步任务时使用)',
    `used_memory_ids`      json                                                            NULL COMMENT '本次注入引用的记忆ID列表(JSON数组,注入可见性)',
    `edited`               tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否已编辑(0:否;1:是，编辑重发后原消息标记)',
    `original_content`     LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL COMMENT '编辑前原文(edited=1时填充，支撑已编辑标识和编辑历史)',
    `deleted`              tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`            bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`            bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`          datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`          datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_conversation_time` (`conversation_id`, `create_time`) USING BTREE,
    INDEX `idx_parent` (`parent_message_id`) USING BTREE,
    INDEX `idx_task` (`task_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话消息表'
  ROW_FORMAT = DYNAMIC;
