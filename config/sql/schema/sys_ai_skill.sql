-- ============================================================
-- 表名: sys_ai_skill
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- Skill 主表（F-M08-006 Skills 管理），承载 Skill 元数据与 Markdown 指令全文，
-- 与渐进式加载的关系：SkillManager 启动时经 refresh_index 仅加载启用项的
--   名称+描述进内存索引（几十 tokens），LLM 判断需要时通过 skill_load 工具
--   从内存缓存读取完整指令，避免无关 Skill 挤占上下文。
-- 与 sys_ai_agent_skill 的关联语义：sys_ai_agent_skill.skill_name 即本表
--   name 的外键语义（项目惯例不加物理外键）；删除 Skill 前须校验其是否被
--   Agent 关联，有则拒绝并提示先解绑。
-- name 业务唯一（类别②：绕过软删查全表判重，删除后不可复用）。
-- status 标识 Skill 启停（1=启用，2=禁用），禁用后不再进入 SkillManager 索引，
--   从而不出现在 discover/load 返回中。
-- source 标记来源（builtin=内置播种，admin=管理员创建）。
-- 配置类表，使用逻辑删除（SoftDeleteMixin），deleted 由全局 do_orm_execute 过滤。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_skill`;
CREATE TABLE `sys_ai_skill`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT 'Skill名称(唯一,关联sys_ai_agent_skill.skill_name)',
    `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT 'Skill描述(供LLM渐进式加载索引使用)',
    `content`     text                                                            NULL COMMENT 'Markdown指令全文(skill_load时完整注入)',
    `status`      tinyint                                                         NOT NULL DEFAULT 1 COMMENT '启停状态(1:启用;2:禁用)',
    `source`      varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL DEFAULT 'admin' COMMENT '来源(builtin:内置播种;admin:管理员创建)',
    `deleted`     tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_name` (`name`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI对话Skill主表(F-M08-006)'
  ROW_FORMAT = DYNAMIC;
