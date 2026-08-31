-- ============================================================
-- 种子数据: sys_ai_model（AI 模型配置表）
-- 模块: 基础模块-AI模型管理
-- ============================================================
-- 正式模型清单：
--   chat:      gpt-4o-mini(openai) / qwen3-0.6b(内置本地) / deepseek-v4-flash(deepseek)
--   embedding: bge-m3(内置本地, 1024维) —— 与后端 EMBEDDING_MODEL_VALUES 白名单、
--              本地向量服务(8992 /v1/embeddings 接受任意模型名)及 SDK 测试口径一致；
--              本地向量模型真实标识 qwen3-embedding-0.6b 由后端 ensure_local_models
--              幂等播种为停用注册记录(status=0)，不在此清单。
-- 说明：
--   1. 本文件由 rebuild_mysql.py --import 全量同步（先清表再导入），
--      历史测试残留（SDK 集成测试创建的 test_* 模型）不在此清单内。
--   2. id 显式固定：deepseek-v4-flash.fallback_model_id=70 引用 qwen3-0.6b，
--      固定 id 保证降级引用有效；embedding 用新 id=200 避免与既有自增冲突。
--   3. rerank 模型本地暂无能力，不配置伪模型，知识库 rerank 保持关闭。
-- ============================================================
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO `sys_ai_model` (`id`,`provider_id`,`model_id`,`model_type`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_multimodal`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`supports_structured_output`,`extra_request_params`,`fallback_model_id`,`prompt_cache_prefix_len`,`image_tokens_per_image`,`concurrency_limit`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('1','1','gpt-4o-mini','chat',NULL,'GPT-4o mini','128000','4096','1','1','1','1','1',NULL,NULL,'0','0',NULL,'1','0','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_model` (`id`,`provider_id`,`model_id`,`model_type`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_multimodal`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`supports_structured_output`,`extra_request_params`,`fallback_model_id`,`prompt_cache_prefix_len`,`image_tokens_per_image`,`concurrency_limit`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('70','77','qwen3-0.6b','chat',NULL,'Qwen3-0.6B（内置本地）','16384','2048','0','1','1','0','0',NULL,NULL,'0','0',NULL,'1','0','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_model` (`id`,`provider_id`,`model_id`,`model_type`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_multimodal`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`supports_structured_output`,`extra_request_params`,`fallback_model_id`,`prompt_cache_prefix_len`,`image_tokens_per_image`,`concurrency_limit`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('169','177','deepseek-v4-flash','chat',NULL,'DeepSeek V4 Flash','128000','8192','0','1','1','1','0',NULL,'70','0','0',NULL,'1','0','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_model` (`id`,`provider_id`,`model_id`,`model_type`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_multimodal`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`supports_structured_output`,`extra_request_params`,`fallback_model_id`,`prompt_cache_prefix_len`,`image_tokens_per_image`,`concurrency_limit`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)  VALUES ('200','77','bge-m3','embedding','1024','BGE-M3（内置本地向量）','8192','2048','0','0','0','0','0',NULL,NULL,'0','0',NULL,'1','0','0','2','2','2026-08-21 00:00:00','2026-08-21 00:00:00');

SET FOREIGN_KEY_CHECKS = 1;
