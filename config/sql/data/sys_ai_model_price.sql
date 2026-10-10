-- ============================================================
-- 种子数据: sys_ai_model_price / sys_ai_model_price_detail（AI 模型用户售价，版本主表 + 档位明细）
-- 模块: 核心模块-AI对话 / 基础模块-AI模型管理
-- ============================================================
-- 第三方模型（火山引擎方案）用户售价，1 元 = 100 积分：
--   * chat  doubao-seed-2-0-mini-260428（方舟低优，provider_id=180）
--       billing_unit=million_tokens，按输入长度分三档；无峰谷差价（peak/idle 同价），
--       输入 10 / 缓存命中 4 / 输出 100 积分（0~32k 档），(32k,128k] 20/4/200，(128k,256k] 40/4/400
--   * asr   bigmodel（火山，provider_id=181）billing_unit=second，1 元/小时 → 0.0278 积分/秒
--   * tts   volc-tts-2-0（火山，provider_id=181）billing_unit=ten_thousand_chars，3 元/万字符 → 300 积分/万字符
-- 说明：
--   1. 免费模型不登记售价：硅基流动 embedding（BAAI/bge-m3）/ rerank（BAAI/bge-reranker-v2-m3）为免费额度，
--      按"未配置价格 = 不扣费 + 仍落账（credits=0）"口径处理，故不落价格行。
--   2. 非 token 模型单行 unit 档（usage_kind='unit'，token_type 复用 'input'，不设峰谷）；
--      计费维度由主表 billing_unit 承载，unit 列非计费依据，统一取 schema 默认。
--   3. 幂等：本文件只管理 id 900~902 三个价格版本及其档位，导入前先按 id 清两表，
--      配合 rebuild_mysql.py --import（先清主表）重复执行不报错、不产生重复行。
--   4. **仅供全新库为主**：本文件虽自带 DELETE 自清理（`--import` 可重复执行），但价格行的
--      `provider_id` 需与供应商实际 id 对应；存量库（dev/pre）供应商 id 可能与 data 里的固定小号
--      不一致（被测试造号占用），故存量环境统一走幂等 migration
--      `config/sql/migration/2026-10-01_seed_volcano_providers_models.sql`（按 provider_code 反查）。
-- ============================================================
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DELETE FROM `sys_ai_model_price_detail` WHERE `price_id` IN (900, 901, 902);
DELETE FROM `sys_ai_model_price` WHERE `id` IN (900, 901, 902);

-- chat：方舟 doubao-seed-2-0-mini-260428（低优档，定价按输入总长度分三档）
INSERT INTO `sys_ai_model_price` (`id`,`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('900','doubao-seed-2-0-mini-260428','180','1','credits_per_million','million_tokens','2026-10-01 00:00:00',NULL,'1','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9001','900','input','token','peak','0','32768','10.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9002','900','input','token','idle','0','32768','10.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9003','900','input','token','peak','32768','131072','20.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9004','900','input','token','idle','32768','131072','20.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9005','900','input','token','peak','131072','262144','40.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9006','900','input','token','idle','131072','262144','40.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9007','900','cached','token','peak','0','32768','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9008','900','cached','token','idle','0','32768','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9009','900','cached','token','peak','32768','131072','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9010','900','cached','token','idle','32768','131072','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9011','900','cached','token','peak','131072','262144','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9012','900','cached','token','idle','131072','262144','4.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9013','900','output','token','peak','0','32768','100.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9014','900','output','token','idle','0','32768','100.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9015','900','output','token','peak','32768','131072','200.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9016','900','output','token','idle','32768','131072','200.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9017','900','output','token','peak','131072','262144','400.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9018','900','output','token','idle','131072','262144','400.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');

-- asr：火山 bigmodel（1 元/小时 → 0.0278 积分/秒）；非 token 单行 unit 档
INSERT INTO `sys_ai_model_price` (`id`,`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('901','bigmodel','181','1','credits_per_million','second','2026-10-01 00:00:00',NULL,'1','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9019','901','input','unit','peak','0',NULL,'0.0278','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');

-- tts：火山 volc-tts-2-0（3 元/万字符 → 300 积分/万字符）；非 token 单行 unit 档
INSERT INTO `sys_ai_model_price` (`id`,`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('902','volc-tts-2-0','181','1','credits_per_million','ten_thousand_chars','2026-10-01 00:00:00',NULL,'1','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_model_price_detail` (`id`,`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('9020','902','input','unit','peak','0',NULL,'300.0000','0','2','2','2026-10-01 00:00:00','2026-10-01 00:00:00');

SET FOREIGN_KEY_CHECKS = 1;
