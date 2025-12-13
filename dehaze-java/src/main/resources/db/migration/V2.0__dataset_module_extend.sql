-- =====================================================
-- 数据集管理模块表结构扩展
-- 版本: V2.0
-- 日期: 2025-12-07
-- 说明: 扩展sys_dataset和sys_item_file表，支持数据集统计、图片标注等功能
-- =====================================================

-- 1. 扩展 sys_dataset 表：新增使用次数统计字段
ALTER TABLE sys_dataset
ADD COLUMN usage_count BIGINT DEFAULT 0 COMMENT '使用次数';

-- 2. 扩展 sys_item_file 表：新增图片元数据和标注字段
ALTER TABLE sys_item_file
ADD COLUMN scene_type VARCHAR(64) DEFAULT NULL COMMENT '场景类型',
ADD COLUMN haze_level VARCHAR(32) DEFAULT NULL COMMENT '雾霾程度',
ADD COLUMN width INT DEFAULT NULL COMMENT '图片宽度',
ADD COLUMN height INT DEFAULT NULL COMMENT '图片高度',
ADD COLUMN usage_count BIGINT DEFAULT 0 COMMENT '使用次数',
ADD COLUMN create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
ADD COLUMN update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间';
