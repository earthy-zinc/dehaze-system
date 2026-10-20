-- ============================================================
-- 2026-09-30 sys_ai_provider 新增 request_timeout_seconds（请求超时能力字段）
-- 权威设计：clarity-doc/docs/03-模块设计/基础模块/AI模型管理/后端实现.md
-- 背景：embedding 调用原先按 provider_code=='local' 特判 300s 超时；改造后由供应商行
--       显式配置（NULL=调用方默认：chat 120s / embedding 30s）。本地 CPU 推理慢，
--       存量内置供应商 local 置 300s，与旧行为等价。
-- 执行方式：mysql --default-character-set=utf8mb4 <本文件>
-- 执行目标：clarity_dev（clarity_test 由 conftest 按 schema+data 全量重建，不跑本文件）
-- 幂等：列已存在时跳过 ADD（information_schema 判定）；UPDATE 仅在 NULL 时补齐
-- 回滚：ALTER TABLE sys_ai_provider DROP COLUMN request_timeout_seconds;
-- ============================================================
SET NAMES utf8mb4;

SET @col_exists = (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'sys_ai_provider'
      AND COLUMN_NAME = 'request_timeout_seconds'
);
SET @sql = IF(
    @col_exists = 0,
    'ALTER TABLE `sys_ai_provider` ADD COLUMN `request_timeout_seconds` int NULL DEFAULT NULL COMMENT ''请求超时(秒;NULL=调用方默认:chat 120/embedding 30);本地 CPU 推理等慢供应商应显式配置(如 300)'' AFTER `health_check_enabled`',
    'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 存量内置供应商 local（CPU 推理慢）补齐 300s；第三方供应商保持 NULL（用调用方默认）
UPDATE `sys_ai_provider`
SET `request_timeout_seconds` = 300
WHERE `provider_code` = 'local'
  AND `deleted` = 0
  AND `request_timeout_seconds` IS NULL;
