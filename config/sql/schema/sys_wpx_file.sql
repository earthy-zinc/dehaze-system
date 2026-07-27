-- ============================================================
-- 表名: sys_wpx_file
-- 模块: 扩展模块
-- ============================================================
-- 设计思路:
-- WPX 数据集文件映射表，记录原始文件与 WPX 格式新文件的对应关系。
-- origin_md5 和 new_md5 双唯一索引，确保映射关系一一对应。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_wpx_file`;
CREATE TABLE `sys_wpx_file`
(
    `id`             bigint          NOT NULL AUTO_INCREMENT COMMENT 'id',
    `origin_file_id` bigint COMMENT '旧文件id',
    `origin_md5`     char(32) unique NOT NULL COMMENT '旧文件的MD5值',
    `origin_path`    varchar(255)    NOT NULL COMMENT '旧文件路径',
    `new_file_id`    bigint COMMENT '新文件id',
    `new_path`       varchar(255)    NOT NULL COMMENT '新文件路径',
    `new_md5`        char(32) unique NOT NULL COMMENT '新文件的MD5值',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_origin_md5` (`origin_md5`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='WPX文件表';
