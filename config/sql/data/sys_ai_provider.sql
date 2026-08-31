-- ============================================================
-- 种子数据: sys_ai_provider（AI 模型供应商配置表）
-- 模块: 基础模块-AI模型管理
-- ============================================================
-- 正式供应商清单（固定 id，与 sys_ai_model.sql 的 provider_id 引用一一对应）：
--   id=1   openai    OpenAI         https://api.openai.com/v1
--   id=77  local     内置本地模型    http://127.0.0.1:8992/v1（内置本地 LLM 子进程服务）
--   id=177 deepseek  DeepSeek       https://api.deepseek.com
-- 说明：
--   1. 本文件为供应商数据的单一信息源，由 rebuild_mysql.py --import 全量同步
--      （先清表再导入），历史测试残留（SDK 集成测试创建的 test_prov_* 供应商）
--      不在清单内，全量同步后即被清除。
--   2. id 显式固定：sys_ai_model.sql 的 provider_id（1/77/177）与 sys_ai_provider_key
--      的 provider_id 均引用此三值，固定 id 保证引用不失效。
--   3. API Key 不在此播种：key_hash/key_cipher 涉及密钥，由管理页面或运行时
--      ensure_local_models 播种，不进入版本库。
--   4. user_identity_forward 全部为 NULL（不启用），需透传时由管理页面按供应商能力配置。
-- ============================================================
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('1','openai','OpenAI','https://api.openai.com/v1','openai_compat','bearer',NULL,'0','1',NULL,NULL,'1','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('77','local','内置本地模型','http://127.0.0.1:8992/v1','openai_compat','bearer',NULL,'0','1',NULL,'内置本地 LLM 服务（127.0.0.1:8992），Key 为占位值','1','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('177','deepseek','DeepSeek（外部真实模型）','https://api.deepseek.com','openai_compat','bearer',NULL,'10','1',NULL,'真实外部模型测试接入','1','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');

SET FOREIGN_KEY_CHECKS = 1;
