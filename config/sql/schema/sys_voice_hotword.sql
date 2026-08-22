-- ============================================================
-- 表名: sys_voice_hotword
-- 模块: 基础模块-语音交互
-- ============================================================
-- 设计思路:
-- ASR 领域热词表，提升专业术语识别率（F-VS-004）。
-- scope 区分作用域：global(全局，所有用户生效，管理员维护)/user(用户级，仅本人生效)。
-- user_id 仅用户级热词填写，global 时为 NULL；内容存储前经 XSS 转义（防热词注入）。
-- 单用户热词上限 100 由应用层校验；软删除，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_voice_hotword`;
CREATE TABLE `sys_voice_hotword`
(
    `id`          bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `word`        varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '热词内容(XSS转义后存储)',
    `scope`       varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL DEFAULT 'user' COMMENT '作用域(global:全局;user:用户级)',
    `user_id`     bigint                                                          NULL DEFAULT NULL COMMENT '所属用户ID(关联sys_user.id，global时为NULL)',
    `create_by`   bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time` datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                        NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_scope_user_deleted` (`scope`, `user_id`, `deleted`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '语音热词表'
  ROW_FORMAT = DYNAMIC;
