-- ============================================================
-- 表名: sys_algorithm_version
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 算法版本历史表，每次发布新版本时保存配置快照。
-- uk_algo_version (algorithm_id, version) 唯一索引防止同算法版本号重复。
-- is_active 标记当前活跃版本，同一算法仅一个活跃版本。
-- config_json 存储完整配置快照，支持版本回滚时恢复。
-- ------------------------------------------------------------

DROP TABLE IF EXISTS `sys_algorithm_version`;
CREATE TABLE `sys_algorithm_version`
(
    `id`           bigint      NOT NULL AUTO_INCREMENT COMMENT '主键',
    `algorithm_id` bigint      NOT NULL COMMENT '关联算法ID',
    `version`      varchar(50) NOT NULL COMMENT '版本号',
    `change_log`   TEXT        NULL COMMENT '变更日志',
    `status`       int         NULL COMMENT '该版本时的状态',
    `config_json`  TEXT        NULL COMMENT '该版本时的配置JSON',
    `model_file_id` bigint     NULL COMMENT '模型文件ID',
    `is_active`    tinyint(1)  NULL DEFAULT 0 COMMENT '是否当前活跃版本',
    `create_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime    NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint      NULL COMMENT '创建人ID',
    `update_by`    bigint      NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_algo_version` (`algorithm_id`, `version`) USING BTREE,
    INDEX `idx_algorithm_id` (`algorithm_id`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  ROW_FORMAT = DYNAMIC COMMENT ='算法版本历史表';
