-- ============================================================
-- 表名: sys_wpx_file
-- 模块: 扩展模块
-- ============================================================
-- 设计思路:
-- WPX 数据集文件映射表，记录原始文件与 WPX 格式新文件的对应关系。
-- origin_md5 和 new_md5 双唯一索引，确保映射关系一一对应。
-- 文件映射为只追加记录，不使用逻辑删除。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_wpx_file`;
CREATE TABLE `sys_wpx_file`
(
    `id`             bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `origin_file_id` bigint                                                         NULL DEFAULT NULL COMMENT '旧文件id',
    `origin_md5`     char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NOT NULL COMMENT '旧文件的MD5值',
    `origin_path`    varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '旧文件路径',
    `new_file_id`    bigint                                                         NULL DEFAULT NULL COMMENT '新文件id',
    `new_path`       varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '新文件路径',
    `new_md5`        char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NOT NULL COMMENT '新文件的MD5值',
    `create_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_origin_md5` (`origin_md5`) USING BTREE,
    UNIQUE INDEX `uk_new_md5` (`new_md5`) USING BTREE,
    INDEX `idx_origin_md5` (`origin_md5`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'WPX文件表'
  ROW_FORMAT = DYNAMIC;
