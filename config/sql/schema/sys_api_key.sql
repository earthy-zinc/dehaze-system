-- ============================================================
-- 表名: sys_api_key
-- 模块: 扩展模块
-- ============================================================
-- 设计思路:
-- API 密钥表，存储用户创建的长期鉴权密钥。
-- 密钥明文仅在创建时返回一次，数据库存储 SHA-256 哈希值（key_hash）。
-- key_prefix 用于前端展示识别（如 dhak_ab3x），不参与鉴权。
-- 吊销机制：使用 revoked_at 字段（NULL=未吊销，非NULL=已吊销且记录吊销时间）。
--   本表不使用逻辑删除（无 deleted 字段）：API Key 唯一的"移除"即吊销，
--   吊销后 hash 必须永久保留以拒绝已泄露的旧密钥，故用 revoked_at 标记而非删除。
-- 被吊销的 key 永久保留 hash，绝不物理删、绝不复用，确保无法否认已泄露旧密钥。
-- 三端（Java/Go/Python）共享此表，同一密钥可在任意后端通过 Authorization: Bearer 鉴权。
-- Key 级配额与模型白名单：daily_quota/monthly_quota/rpm_limit 与 model_whitelist 均 NULL=不限制，
--   计数走 Redis（apikey:{key_id}:daily:{date} 等），与用户积分配额双轨控制；
--   model_whitelist 为 NULL 或空数组 = 继承用户可见模型（不拦截）。
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
    `expires_at`   datetime     NULL DEFAULT NULL COMMENT '过期时间(NULL表示永不过期)',
    `last_used_at` datetime     NULL DEFAULT NULL COMMENT '最后使用时间',
    `revoked_at`   datetime     NULL DEFAULT NULL COMMENT '吊销时间(NULL表示未吊销)',
    `daily_quota`  bigint       NULL DEFAULT NULL COMMENT '日调用配额(NULL或0表示不限制)',
    `monthly_quota` bigint      NULL DEFAULT NULL COMMENT '月调用配额(NULL或0表示不限制)',
    `rpm_limit`    int          NULL DEFAULT NULL COMMENT '每分钟请求数上限RPM(NULL或0表示不限制)',
    `model_whitelist` json      NULL DEFAULT NULL COMMENT '模型白名单(NULL或空数组表示继承用户可见模型)',
    `create_time`  datetime     NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint       NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint       NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_key_hash` (`key_hash`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_revoked_at` (`revoked_at`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'API密钥表(吊销用revoked_at,不使用逻辑删除)'
  ROW_FORMAT = DYNAMIC;
