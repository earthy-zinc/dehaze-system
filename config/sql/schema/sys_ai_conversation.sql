-- ============================================================
-- 表名: sys_ai_conversation
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 对话会话表，记录用户与 LLM 的会话上下文容器。
-- model 关联 sys_ai_model.model_id，记录会话使用的模型，用户切换模型时更新。
-- agent_code 关联 sys_ai_agent.agent_code，记录会话使用的智能体（系统提示词/推理范式/工具集等），
--   为空时使用平台默认 Agent（agent_code='default'）；切换 Agent 后下一条消息生效。
-- agent_version 记录会话锚定的 Agent 已发布版本号（创建/切换会话时写入，发布/回滚不影响进行中会话，
--   运行面据此读取不可变快照组装，保证行为可复现）。
-- summary 存储超 token 阈值时自动生成的老消息摘要，支撑长对话上下文压缩。
-- summary_upto_message_id 为摘要水位（已覆盖的最后一条消息 ID），用于增量摘要，
--   只摘要"上次水位之后、最近 N 轮之前"的消息，避免全量重摘导致摘要无限膨胀。
-- system_prompt 存储会话级系统提示词，不同场景（去雾助手 vs 通用问答）可配置不同 prompt。
-- model_config(JSON) 存储 temperature、top_p、max_tokens 等模型参数，与会话绑定。
-- api_key_id 关联 sys_api_key，MCP 工具调用时用此 Key 向后端透传身份（不存明文）。
-- message_count 冗余消息数，避免 COUNT 查询；last_message_at 用于会话列表按活跃度排序。
-- 多模态视觉读取按用户全局日计数（Redis ai:multimodal:{userId}:{date}），不落会话表。
-- current_branch_message_id 记录当前激活的分支末端消息 ID，支持分支切换后从正确位置继续对话。
-- last_read_message_id 记录用户最后已读消息 ID，支撑多端登录场景下的已读/未读状态同步。
-- title_source 标识标题来源（auto:LLM自动生成;manual:手动修改），title 为空时异步用 LLM 生成。
-- pinned 支持会话置顶，pinned_at 记录置顶时间（置顶会话按此倒序）；置顶上限 10 个由服务层校验。
-- delete_time 记录软删除时间（30 天恢复窗口判定，超期由定时任务物理清理）。
-- status 区分活跃/已归档，删除走 deleted 字段。
-- 会话无业务唯一键（标题可重复），软删除无唯一索引冲突，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_conversation`;
CREATE TABLE `sys_ai_conversation`
(
    `id`                      bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`                 bigint                                                          NOT NULL COMMENT '用户ID',
    `title`                   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT '新对话' COMMENT '会话标题(首条消息自动提取，支持手动修改)',
    `model`                   varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '会话使用的模型标识(关联sys_ai_model.model_id)',
    `agent_code`              varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NULL DEFAULT NULL COMMENT '会话使用的智能体编码(关联sys_ai_agent.agent_code,为空使用默认Agent)',
    `agent_version`           int                                                             NULL DEFAULT NULL COMMENT '会话锚定的Agent已发布版本号(创建/切换会话时写入,发布/回滚不影响进行中会话)',
    `summary`                 TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '会话摘要(超token阈值时自动摘要老消息)',
    `summary_upto_message_id` bigint                                                          NULL DEFAULT NULL COMMENT '摘要水位：已纳入摘要覆盖范围的最后一条消息ID(增量摘要推进依据)',
    `system_prompt`           TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '系统提示词(会话级，不同场景可配置不同prompt)',
    `model_config`            json                                                            NULL COMMENT '模型参数配置(temperature;top_p;max_tokens等)',
    `suggestions_enabled`     tinyint                                                         NOT NULL DEFAULT 1 COMMENT '类似问题推荐开关(0:关;1:开,关闭后回复完成不推送suggestions事件)',
    `api_key_id`              bigint                                                          NULL DEFAULT NULL COMMENT '绑定的API Key ID(关联sys_api_key，MCP工具调用身份透传)',
    `message_count`           int                                                             NOT NULL DEFAULT 0 COMMENT '消息数(冗余计数，避免COUNT查询)',
    `last_message_at`         datetime                                                        NULL DEFAULT NULL COMMENT '最后消息时间(会话列表按此排序)',
    `current_branch_message_id` bigint                                                        NULL DEFAULT NULL COMMENT '当前激活的分支末端消息ID(分支切换后从此位置继续对话)',
    `last_read_message_id`    bigint                                                          NULL DEFAULT NULL COMMENT '最后已读消息ID(多端已读未读状态同步)',
    `pinned`                  tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否置顶(0:否;1:是)',
    `pinned_at`               datetime                                                        NULL DEFAULT NULL COMMENT '置顶时间(置顶会话按此倒序)',
    `delete_time`             datetime                                                        NULL DEFAULT NULL COMMENT '软删时间(30天恢复窗口判定，超期由定时任务物理清理)',
    `title_source`            varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'auto' COMMENT '标题来源(auto:LLM自动生成;manual:手动修改)',
    `status`                  tinyint                                                         NOT NULL DEFAULT 1 COMMENT '会话状态(1:活跃;2:已归档)',
    `deleted`                 tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`               bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`               bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`             datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`             datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_list` (`user_id`, `deleted`, `last_message_at` DESC) USING BTREE,
    INDEX `idx_user_pinned` (`user_id`, `pinned`, `pinned_at` DESC) USING BTREE,
    INDEX `idx_deleted_time` (`deleted`, `delete_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话会话表'
  ROW_FORMAT = DYNAMIC;
