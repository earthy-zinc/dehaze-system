-- ============================================================
-- 表名: sys_file
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 文件元数据表，通过 MD5 唯一索引实现文件去重，相同内容只存一份。
-- size_bytes 存储原始字节数用于精确计算，size 存储格式化字符串用于展示。
-- object_name 是存储后端中的对象键（与环境无关）；storage 标识存储后端（minio/local/nginx-static）。
-- URL 永远运行时拼接（storage.baseUrl + object_name），不落库，环境迁移只改配置。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_file`;
CREATE TABLE `sys_file`
(
    `id`          bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `type`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '文件类型',
    `name`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件原始名',
    `object_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '对象键（存储后端中的定位，与环境无关）',
    `storage`     varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'minio' COMMENT '存储后端标识(minio/local/nginx-static)',
    `size`        varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '0' COMMENT '文件大小（格式化显示）',
    `size_bytes`  bigint                                                         NULL DEFAULT NULL COMMENT '文件大小（原始字节数）',
    `md5`         char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NOT NULL COMMENT '文件的MD5值，用于比对文件是否相同',
    `deleted`     tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_md5` (`md5` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '文件表'
  ROW_FORMAT = DYNAMIC;
