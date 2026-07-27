-- ============================================================
-- 表名: sys_file
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 文件元数据表，通过 MD5 唯一索引实现文件去重，相同内容只存一份。
-- size_bytes 存储原始字节数用于精确计算，size 存储格式化字符串用于展示。
-- object_name 是 MinIO 对象名，path 是逻辑路径，url 是访问地址。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_file`;
CREATE TABLE `sys_file`
(
    `id`          int             NOT NULL AUTO_INCREMENT COMMENT '文件id',
    `type`        varchar(100)             DEFAULT NULL COMMENT '文件类型',
    `url`         TEXT                     DEFAULT NULL COMMENT '文件url',
    `name`        varchar(100)    NOT NULL COMMENT '文件原始名',
    `object_name` varchar(100)    NOT NULL COMMENT '文件存储名',
    `size`        varchar(100)    NOT NULL DEFAULT '0' COMMENT '文件大小（格式化显示）',
    `size_bytes`  bigint                   DEFAULT NULL COMMENT '文件大小（原始字节数）',
    `path`        varchar(255)    NOT NULL COMMENT '文件路径',
    `md5`         char(32) UNIQUE NOT NULL COMMENT '文件的MD5值，用于比对文件是否相同',
    `create_time` datetime        NOT NULL COMMENT '创建时间',
    `update_time` datetime                 DEFAULT NULL COMMENT '更新时间',
    `create_by`   bigint                   DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                   DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `md5_key` (`md5` ASC) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='文件表';
