-- 算法版本历史表
CREATE TABLE IF NOT EXISTS sys_algorithm_version (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    algorithm_id BIGINT NOT NULL COMMENT '关联算法ID',
    version VARCHAR(50) NOT NULL COMMENT '版本号',
    change_log TEXT COMMENT '变更日志',
    status INT COMMENT '该版本时的状态',
    config_json TEXT COMMENT '该版本时的配置JSON',
    model_file_id BIGINT COMMENT '模型文件ID',
    is_active TINYINT(1) DEFAULT 0 COMMENT '是否当前活跃版本',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    create_by BIGINT,
    update_by BIGINT,
    UNIQUE KEY uk_algo_version (algorithm_id, version),
    INDEX idx_algorithm_id (algorithm_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='算法版本历史表';

-- 图像输入历史记录表
CREATE TABLE IF NOT EXISTS sys_input_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT '用户ID',
    original_image_url VARCHAR(500) COMMENT '原始图片URL',
    original_thumbnail_url VARCHAR(500) COMMENT '原始缩略图URL',
    result_image_url VARCHAR(500) COMMENT '处理结果图片URL',
    result_thumbnail_url VARCHAR(500) COMMENT '结果缩略图URL',
    algorithm_id BIGINT COMMENT '算法ID',
    algorithm_name VARCHAR(100) COMMENT '算法名称（冗余）',
    algorithm_params TEXT COMMENT '算法参数（JSON）',
    processing_time INT COMMENT '处理耗时（毫秒）',
    status TINYINT DEFAULT 3 COMMENT '处理状态（1=成功，2=失败，3=处理中）',
    input_source VARCHAR(20) COMMENT '图片来源（upload/camera/sample）',
    is_favorite TINYINT(1) DEFAULT 0 COMMENT '是否收藏',
    sync_status TINYINT DEFAULT 0 COMMENT '同步状态（0=未同步，1=已同步）',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    create_by BIGINT,
    update_by BIGINT,
    INDEX idx_user_time (user_id, create_time DESC),
    INDEX idx_user_favorite (user_id, is_favorite, create_time DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='图像输入历史记录表';

-- sys_algorithm 表新增字段
ALTER TABLE sys_algorithm
    ADD COLUMN IF NOT EXISTS version VARCHAR(50) DEFAULT NULL COMMENT '算法版本号' AFTER type,
    ADD COLUMN IF NOT EXISTS audit_by BIGINT DEFAULT NULL COMMENT '审核人ID' AFTER status,
    ADD COLUMN IF NOT EXISTS audit_time DATETIME DEFAULT NULL COMMENT '审核时间' AFTER audit_by,
    ADD COLUMN IF NOT EXISTS audit_remark VARCHAR(500) DEFAULT NULL COMMENT '审核备注' AFTER audit_time;
