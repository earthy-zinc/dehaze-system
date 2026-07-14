-- Problem 3.5: 统一 sys_task 表审计字段命名（created_by → create_by），补全 update_by 字段
-- 与其他表（sys_user/sys_role/sys_menu/sys_dict/sys_dict_type/sys_dept/sys_dataset/sys_algorithm 等）保持一致

-- 1. 重命名列 created_by → create_by（MySQL 8.0+ 支持 RENAME COLUMN，避免数据丢失）
ALTER TABLE sys_task
    RENAME COLUMN created_by TO create_by;

-- 2. 新增 update_by 字段（与其它表对齐）
ALTER TABLE sys_task
    ADD COLUMN IF NOT EXISTS update_by BIGINT NULL COMMENT '修改人ID' AFTER create_by;

-- 3. 重建索引（idx_created_by → idx_create_by）
ALTER TABLE sys_task
    DROP INDEX IF EXISTS idx_created_by;
ALTER TABLE sys_task
    ADD INDEX IF NOT EXISTS idx_create_by (create_by) USING BTREE;
