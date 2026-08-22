-- ============================================================
-- 表名: sys_ai_agent
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 智能体配置表，管理可配置的 Agent（系统提示词、模型、推理范式、参数、权限规则等）。
-- agent_code 为业务引用键（如 default、image-analyst），唯一索引，删除后不可复用（类别②）。
-- 推理引擎基于 deepagents（LangChain 官方 Agent Harness），DeepAgentBuilder 从本表加载配置
--   并组装 create_deep_agent 入参，返回编译后的 CompiledStateGraph。
-- system_prompt 存 Agent 的指令（Markdown），为空时由 deepagents 使用内置默认提示。
-- model_id 关联 sys_ai_model.model_id（字符串引用键，非主键），标识 Agent 使用的 LLM。
-- reasoning_mode 标识推理范式：auto(复杂度评估自动选择)/direct/react/plan_execute/reflexion。
-- config(JSON) 存推理参数（max_steps/token_budget/max_parallel/tool_timeout/retry_max/
--   reflexion_threshold/temperature 等），为空的键继承 sys_dict 系统默认（ai_reasoning_defaults）。
-- is_subagent 标识是否可作为子 Agent（被其他 Agent 通过 task 工具调用），不可被会话直接选择。
-- is_team 标识是否为 Team 团队（通过 langgraph-supervisor 编排多 Agent 协作）。
--   is_subagent 和 is_team 互斥（普通 Agent 两者均为 0）。
-- is_exposed 标识是否对外暴露为 A2A 子 Agent（默认不暴露，安全默认值；仅启用且非子 Agent
--   的普通 Agent 可暴露，子 Agent 不可独立暴露，见 §5.4）。
-- permissions(JSON) 存 deepagents FilesystemPermission 权限规则（operations/paths/mode），
--   控制虚拟文件系统的读写范围；mode 支持 allow/deny/interrupt，interrupt 触发用户确认。
-- 配置类表，使用逻辑删除；agent_code 为业务引用键，删除后不可复用（类别②，查重查全表）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_agent`;
CREATE TABLE `sys_ai_agent`
(
    `id`              bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `agent_code`      varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT 'Agent唯一编码(业务引用键,如default;image-analyst)',
    `name`            varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT 'Agent显示名称',
    `description`     varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT 'Agent描述(供LLM决策调用时参考)',
    `system_prompt`   TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '系统提示词(Markdown,为空时由deepagents使用内置默认)',
    `model_id`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '关联模型标识(关联sys_ai_model.model_id)',
    `reasoning_mode`  varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'auto' COMMENT '推理范式(auto:复杂度评估自动选择;direct:直接回复;react:边想边做;plan_execute:先想后做;reflexion:反思迭代)',
    `config`          json                                                            NULL COMMENT '推理参数JSON(max_steps/token_budget/max_parallel/tool_timeout/retry_max/reflexion_threshold/temperature等,为空继承sys_dict系统默认)',
    `is_subagent`     tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否可作为子Agent(0:否;1:是,不可被会话直接选择)',
    `is_team`         tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否为Team团队(0:否;1:是,通过langgraph-supervisor编排多Agent协作)',
    `is_exposed`      tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否对外暴露为A2A子Agent(0:否;1:是,默认不暴露;仅启用且非子Agent的普通Agent可暴露)',
    `permissions`     json                                                            NULL COMMENT '文件系统权限规则JSON(deepagents FilesystemPermission:operations/paths/mode,mode支持allow/deny/interrupt)',
    `sort_order`      int                                                             NOT NULL DEFAULT 0 COMMENT '排序序号(数字越小越靠前)',
    `status`          tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`         tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`       bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_agent_code` (`agent_code`) USING BTREE,
    INDEX `idx_model` (`model_id`) USING BTREE,
    INDEX `idx_status_type` (`status`, `is_subagent`, `is_team`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI智能体配置表'
  ROW_FORMAT = DYNAMIC;
