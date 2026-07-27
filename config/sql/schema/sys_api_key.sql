-- ============================================================
-- 表名: sys_api_key
-- 模块: 扩展模块
-- ============================================================
-- 设计思路:
-- API 密钥表，存储用户创建的长期鉴权密钥。
-- 密钥明文仅在创建时返回一次，数据库存储 SHA-256 哈希值（key_hash）。
-- key_prefix 用于前端展示识别（如 dhak_ab3x），不参与鉴权。
-- 三端（Java/Go/Python）共享此表，同一密钥可在任意后端通过 Authorization: Bearer 鉴权。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_api_key`;
CREATE TABLE `sys_api_key`
(
    `id`           bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`      bigint       NOT NULL COMMENT '所属用户ID',
    `name`         varchar(128) NOT NULL DEFAULT '' COMMENT '密钥名称',
    `key_prefix`   varchar(16)  NOT NULL COMMENT '密钥前缀(用于识别)',
    `key_hash`     varchar(64)  NOT NULL COMMENT '密钥SHA-256哈希',
    `status`       tinyint      NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `expires_at`   datetime     NULL     DEFAULT NULL COMMENT '过期时间(NULL表示永不过期)',
    `last_used_at` datetime     NULL     DEFAULT NULL COMMENT '最后使用时间',
    `create_time`  datetime     NULL     DEFAULT NULL COMMENT '创建时间',
    `update_time`  datetime     NULL     DEFAULT NULL COMMENT '更新时间',
    `create_by`    bigint       NULL     DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint       NULL     DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_key_hash` (`key_hash`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci COMMENT = 'API密钥表'
  ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
