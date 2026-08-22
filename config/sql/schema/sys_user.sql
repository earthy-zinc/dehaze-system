-- ============================================================
-- 表名: sys_user
-- 模块: 系统管理
-- ============================================================
-- 设计思路:
-- 用户表，username 唯一索引防止重复注册。
-- password 存储 BCrypt 哈希，不存明文。avatar 用 TEXT 存储完整 URL。
-- 逻辑删除（deleted 字段）保留用户数据完整性，关联记录不丢失。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user`
(
    `id`          bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `username`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户名',
    `nickname`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '昵称',
    `gender`      tinyint                                                       NULL DEFAULT 1 COMMENT '性别((1:男;2:女))',
    `password`    varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '密码',
    `dept_id`     bigint                                                        NULL DEFAULT NULL COMMENT '部门ID',
    `avatar`      TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '用户头像',
    `mobile`      varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '联系方式',
    `status`      tinyint                                                       NULL DEFAULT 1 COMMENT '用户状态((1:正常;0:禁用))',
    `email`       varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户邮箱',
    `credits_balance` decimal(12, 2)                                             NOT NULL DEFAULT 0.00 COMMENT 'AI积分余额(充值/赠送增加;扣减减少)',
    `credits_version` int                                                        NOT NULL DEFAULT 0 COMMENT 'AI积分余额乐观锁版本号',
    `deleted`     tinyint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_username` (`username` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户信息表'
  ROW_FORMAT = DYNAMIC;
