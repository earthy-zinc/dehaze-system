-- ============================================================
-- 表名: sys_ai_provider_key
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 供应商 API Key 表，每个供应商可配置多个 Key，实现负载均衡和 Key 轮换。
-- key_hash 为 SHA256 哈希，用于查重（同一 Key 不可重复录入）。
-- key_cipher 为 AES-256 加密的密文，运行时解密获取明文（明文不存储、不进入日志）。
-- key_prefix 为密钥前缀（展示用，如 sk-proj-aB...），不存储完整 Key。
-- priority 为优先级（数字越小越优先），同优先级按 weight 加权随机选取。
-- daily_quota 为供应商侧日调用上限（可选），超出则切换其他 Key；日调用计数走 Redis（ai:provider_key:{key_id}:daily:{date}），不落库。
-- rpm_limit 为分钟调用频率上限（可选，NULL 表示不限制），超限后本轮跳过该 Key；分钟计数走 Redis（ai:provider_key:{key_id}:minute:{yyyyMMddHHmm}）。
-- 失败冷却走 Redis（ai:provider_key:{key_id}:unavailable，TTL 随连续失败次数递增），不落库。
-- last_used_at / last_used_by 异步更新（写入 Redis 缓冲，定时批量刷库），避免高并发频繁写库。
-- Key 管理为状态控制（启用/禁用/过期），不使用逻辑删除（物理删除）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_provider_key`;
CREATE TABLE `sys_ai_provider_key`
(
    `id`            bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `provider_id`   bigint                                                          NOT NULL COMMENT '关联供应商ID(关联sys_ai_provider.id)',
    `name`          varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT 'Key名称(备注,如OpenAI主账号;备用账号)',
    `key_hash`     char(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci      NOT NULL COMMENT '密钥哈希(SHA256 hex,固定64字符,用于查重)',
    `key_prefix`   varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL COMMENT '密钥前缀(展示用,如sk-proj-aB...)',
    `key_cipher`   varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '密钥密文(AES-256-CBC加密后base64编码,运行时解密)',
    `status`       tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `priority`     int                                                             NOT NULL DEFAULT 0 COMMENT '优先级(数字越小越优先)',
    `weight`       int                                                             NOT NULL DEFAULT 1 COMMENT '权重(同优先级按权重加权随机选取)',
    `daily_quota`  int                                                             NULL DEFAULT NULL COMMENT '日调用上限(供应商侧限额,可选)',
    `rpm_limit`    int                                                             NULL DEFAULT NULL COMMENT '分钟调用频率上限(可选,NULL表示不限制)',
    `expires_at`   datetime                                                        NULL DEFAULT NULL COMMENT '过期时间',
    `last_used_at` datetime                                                        NULL DEFAULT NULL COMMENT '最后使用时间',
    `last_used_by` bigint                                                          NULL DEFAULT NULL COMMENT '最后使用的用户ID',
    `create_by`    bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`  datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_key_hash` (`key_hash`) USING BTREE,
    INDEX `idx_provider` (`provider_id`, `status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI供应商API密钥表'
  ROW_FORMAT = DYNAMIC;
