-- ============================================================
-- 2026-09-30 内置供应商默认不参与熔断（health_check_enabled 置 0）
-- 权威设计：clarity-doc/docs/03-模块设计/基础模块/AI模型管理/后端实现.md
-- 背景：健康记录原先在编排侧代码按 provider_code=='local' 特判跳过；改造后改为
--       sys_ai_provider.health_check_enabled 字段驱动（能力配置单一事实源）。
--       为保持既有行为（内置服务瞬时失败不熔断、不阻断后续请求），种子与存量数据均置 0；
--       需要熔断保护的部署可在管理端开启该供应商的"健康检查"。
-- 执行方式：mysql --default-character-set=utf8mb4 <本文件>
-- 执行目标：clarity_dev（clarity_test 由 conftest 按 schema+data 全量重建，不跑本文件）
-- 幂等：仅在仍为开启(1)时置 0；重复执行零副作用，未初始化环境命中 0 行
-- 回滚：UPDATE sys_ai_provider SET health_check_enabled=1
--       WHERE provider_code IN ('local','local-voice','local-modal') AND deleted=0;
-- ============================================================
SET NAMES utf8mb4;

UPDATE `sys_ai_provider`
SET `health_check_enabled` = 0
WHERE `provider_code` IN ('local', 'local-voice', 'local-modal')
  AND `deleted` = 0
  AND `health_check_enabled` = 1;
