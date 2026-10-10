-- ============================================================
-- 存量环境补齐：第三方模型（火山引擎方案）供应商 / 模型 / 售价
-- 模块: 基础模块-AI模型管理
-- ============================================================
-- 【为什么需要本文件】
--   `config/sql/data/*.sql` 是**全新库**用的种子（plain INSERT + 固定主键 + 多值 INSERT 原子性）：
--   全新库（DROP+CREATE 后全量重建）可直接导入；**存量环境**（dev/pre）重跑会因主键/唯一键
--   冲突 `ERROR 1062` 整条中止，后续新增行一条都进不去。故存量环境用本 migration 幂等补齐。
--
-- 【收敛口径：只认业务键，绝不按主键命中】
--   存量库里 180–182 / 500–505 等小号 id 可能已被测试脏数据占用（`test_prov_*` / `test_model_*` /
--   `favmodel_*`）。因此本文件：
--     * **INSERT 列里不写固定主键 `id`**（交由自增分配）——曾用「固定 id + ON DUPLICATE KEY UPDATE」
--       的一版会按**主键冲突**命中无关行并把其列改成火山/硅基的值（已在 pre 复现），本版彻底修正；
--     * 一律 `INSERT ... SELECT ... WHERE NOT EXISTS(业务键)` 守卫：业务键已存在（含管理员改过的）
--       一律**不动**，业务键不匹配的行**一行都不碰**；
--     * `model.provider_id` / `price.provider_id` 按 `provider_code` **反查**，不写死 180/181/182；
--     * `detail.price_id` 按 `(model_id, provider_code, price_version)` 反查，不写死价格主键。
--   业务键：provider `uk_provider_code(provider_code[,deleted])`、
--           model `uk_model_provider(model_id, provider_id, deleted)`、
--           price `uk_model_provider_version(model_id, provider_id, price_version, deleted)`、
--           detail `uk_price_token_time_range(price_id, token_type, time_slot, min_tokens, max_tokens, deleted)`。
--
-- 【执行方式】不随 schema+data 自动导入（`config/sql/migration/` 不在 docker-init 挂载里）：
--     mysql -h<host> -u<user> -p<pass> <db> < 2026-10-01_seed_volcano_providers_models.sql
--   幂等：连跑两次结果一致、无重复行；已有行不被覆盖（`status`/`deleted`/`create_*` 更不会被动）。
--   **唯一例外**：存量 `volc-voice` 行按新版控制台单密钥口径升级 `default_headers`（见下方 UPDATE 段，
--   仅在仍为旧口径时命中，保留管理端已填的 APP ID）。
-- 【不写密钥】`sys_ai_provider_key` 一行不落（APP ID/Access Token/API Key 由管理端加密写入）。
-- 【默认模型不切】本文件不翻转任何 `default_model.*`；切换见 2026-10-01_switch_default_model_volc.sql。
-- 【与 data 种子一致】取值逐列对齐 config/sql/data/{sys_ai_provider,sys_ai_model,sys_ai_model_price}.sql。
-- ============================================================
SET NAMES utf8mb4;

-- ===== 1) 供应商（volc-ark / volc-voice / siliconflow）=====
INSERT INTO `sys_ai_provider`
  (`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)
SELECT 'volc-ark','火山方舟','https://ark.cn-beijing.volces.com/api/v3','openai_compat','bearer',NULL,20,1,NULL,'火山引擎方舟大模型（chat/embedding/rerank）',1,0,2,2,120,NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_ai_provider` WHERE `provider_code`='volc-ark');

INSERT INTO `sys_ai_provider`
  (`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)
SELECT 'volc-voice','火山豆包语音','https://openspeech.bytedance.com','volc_realtime','bearer','{"auth_header":"X-Api-Key","X-Api-App-Key":"","asr_resource_id":"volc.seedasr.sauc.duration","tts_resource_id":"seed-tts-2.0","cluster":"volcano_tts"}',21,1,NULL,'火山引擎流式语音识别/合成（asr/tts 共用供应商；单密钥 X-Api-Key 由管理端加密写入密钥表）',1,0,2,2,60,NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_ai_provider` WHERE `provider_code`='volc-voice');

-- 存量 volc-voice 行口径升级（经典 APP ID+AccessToken → 新版控制台单密钥 X-Api-Key）：
--   上面的 INSERT ... WHERE NOT EXISTS 对已存在行不生效（存量 dev/pre 行已存在），故另补本幂等 UPDATE。
--   只动 `default_headers`（`X-Api-Resource-Id` 拆为能力分列的 `asr_resource_id`/`tts_resource_id`、
--   补 `auth_header=X-Api-Key`），保留管理端已填的 `X-Api-App-Key`(APP ID) 等既有键。
--   守卫：仅在仍是旧口径（带旧键或缺 auth_header）时执行 ⇒ 重复执行结果一致（第二次无行命中）。
UPDATE `sys_ai_provider`
SET `default_headers` = JSON_SET(
      JSON_REMOVE(COALESCE(`default_headers`, JSON_OBJECT()), '$."X-Api-Resource-Id"'),
      '$.auth_header', 'X-Api-Key',
      '$.asr_resource_id', 'volc.seedasr.sauc.duration',
      '$.tts_resource_id', 'seed-tts-2.0'
    )
WHERE `provider_code`='volc-voice' AND `deleted`=0
  AND (
    JSON_EXTRACT(COALESCE(`default_headers`, JSON_OBJECT()), '$."X-Api-Resource-Id"') IS NOT NULL
    OR JSON_EXTRACT(COALESCE(`default_headers`, JSON_OBJECT()), '$.auth_header') IS NULL
  );

INSERT INTO `sys_ai_provider`
  (`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)
SELECT 'siliconflow','硅基流动','https://api.siliconflow.cn/v1','openai_compat','bearer',NULL,22,1,NULL,'硅基流动（chat/embedding/rerank，免费额度）',1,0,2,2,30,NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_ai_provider` WHERE `provider_code`='siliconflow');

-- ===== 2) 模型（provider_id 按 provider_code 反查）=====
-- 方舟 chat
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'doubao-seed-2-0-mini-260428','chat',NULL,NULL,'豆包 Seed 2.0 mini（火山方舟低优）',32768,4096,0,1,1,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-ark' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='doubao-seed-2-0-mini-260428' AND m.provider_id=p.id);

-- 硅基 chat（免费档，默认测试模型候选）
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'Qwen/Qwen3-8B','chat',NULL,NULL,'Qwen3-8B（硅基流动免费档）',32768,8192,1,1,0,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='siliconflow' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='Qwen/Qwen3-8B' AND m.provider_id=p.id);

-- 硅基 embedding
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'BAAI/bge-m3','embedding','["text"]',1024,'BAAI BGE-M3（硅基流动）',4096,4096,0,1,0,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='siliconflow' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='BAAI/bge-m3' AND m.provider_id=p.id);

-- 硅基 rerank
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'BAAI/bge-reranker-v2-m3','rerank',NULL,NULL,'BAAI Reranker v2 M3（硅基流动）',4096,4096,0,1,0,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='siliconflow' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='BAAI/bge-reranker-v2-m3' AND m.provider_id=p.id);

-- 火山 asr
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'bigmodel','asr','["stream"]',NULL,'火山流式语音识别2.0',4096,4096,0,1,0,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-voice' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='bigmodel' AND m.provider_id=p.id);

-- 火山 tts
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'volc-tts-2-0','tts',NULL,NULL,'火山语音合成2.0',4096,4096,0,1,0,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-voice' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id='volc-tts-2-0' AND m.provider_id=p.id);

-- ===== 3) 售价版本（chat 三档 / asr / tts；免费 embedding/rerank 不登记）=====
INSERT INTO `sys_ai_model_price`
  (`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 'doubao-seed-2-0-mini-260428', p.id, 1, 'credits_per_million', 'million_tokens', '2026-10-01 00:00:00', NULL, 1, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-ark' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model_price` pr WHERE pr.model_id='doubao-seed-2-0-mini-260428' AND pr.provider_id=p.id AND pr.price_version=1);

INSERT INTO `sys_ai_model_price`
  (`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 'bigmodel', p.id, 1, 'credits_per_million', 'second', '2026-10-01 00:00:00', NULL, 1, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-voice' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model_price` pr WHERE pr.model_id='bigmodel' AND pr.provider_id=p.id AND pr.price_version=1);

INSERT INTO `sys_ai_model_price`
  (`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 'volc-tts-2-0', p.id, 1, 'credits_per_million', 'ten_thousand_chars', '2026-10-01 00:00:00', NULL, 1, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='volc-voice' AND p.deleted=0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model_price` pr WHERE pr.model_id='volc-tts-2-0' AND pr.provider_id=p.id AND pr.price_version=1);

-- ===== 4) 售价档位明细 =====
-- chat 三档：输入/缓存命中/输出 × 上下文分段 × 峰谷（不分时段，peak/idle 同价）
-- price_id 按 (model_id, provider_code, price_version) 反查；每行按 uk（token_type × 时段 × 分段）守卫。
INSERT INTO `sys_ai_model_price_detail`
  (`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT pr.id, t.token_type, 'token', t.time_slot, t.min_tokens, t.max_tokens, t.unit_price, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_model_price` pr
JOIN `sys_ai_provider` p ON p.id = pr.provider_id AND p.provider_code='volc-ark' AND p.deleted=0
JOIN (
            SELECT 'input'  AS token_type, 'peak' AS time_slot, 0       AS min_tokens, 32768  AS max_tokens, 10.0000 AS unit_price
  UNION ALL SELECT 'input',  'idle',  0,       32768,  10.0000
  UNION ALL SELECT 'input',  'peak',  32768,   131072, 20.0000
  UNION ALL SELECT 'input',  'idle',  32768,   131072, 20.0000
  UNION ALL SELECT 'input',  'peak',  131072,  262144, 40.0000
  UNION ALL SELECT 'input',  'idle',  131072,  262144, 40.0000
  UNION ALL SELECT 'cached', 'peak',  0,       32768,  4.0000
  UNION ALL SELECT 'cached', 'idle',  0,       32768,  4.0000
  UNION ALL SELECT 'cached', 'peak',  32768,   131072, 4.0000
  UNION ALL SELECT 'cached', 'idle',  32768,   131072, 4.0000
  UNION ALL SELECT 'cached', 'peak',  131072,  262144, 4.0000
  UNION ALL SELECT 'cached', 'idle',  131072,  262144, 4.0000
  UNION ALL SELECT 'output', 'peak',  0,       32768,  100.0000
  UNION ALL SELECT 'output', 'idle',  0,       32768,  100.0000
  UNION ALL SELECT 'output', 'peak',  32768,   131072, 200.0000
  UNION ALL SELECT 'output', 'idle',  32768,   131072, 200.0000
  UNION ALL SELECT 'output', 'peak',  131072,  262144, 400.0000
  UNION ALL SELECT 'output', 'idle',  131072,  262144, 400.0000
) t
WHERE pr.model_id='doubao-seed-2-0-mini-260428' AND pr.price_version=1 AND pr.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price_detail` d
    WHERE d.price_id=pr.id AND d.token_type=t.token_type AND d.time_slot=t.time_slot
      AND d.min_tokens <=> t.min_tokens AND d.max_tokens <=> t.max_tokens AND d.deleted=0
  );

-- asr：1 元/小时 → 0.0278 积分/秒；非 token 单行 unit 档
INSERT INTO `sys_ai_model_price_detail`
  (`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT pr.id, 'input', 'unit', 'peak', 0, NULL, 0.0278, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_model_price` pr
JOIN `sys_ai_provider` p ON p.id = pr.provider_id AND p.provider_code='volc-voice' AND p.deleted=0
WHERE pr.model_id='bigmodel' AND pr.price_version=1 AND pr.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price_detail` d
    WHERE d.price_id=pr.id AND d.token_type='input' AND d.time_slot='peak'
      AND d.min_tokens=0 AND d.max_tokens IS NULL AND d.deleted=0
  );

-- tts：3 元/万字符 → 300 积分/万字符；非 token 单行 unit 档
INSERT INTO `sys_ai_model_price_detail`
  (`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT pr.id, 'input', 'unit', 'peak', 0, NULL, 300.0000, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_model_price` pr
JOIN `sys_ai_provider` p ON p.id = pr.provider_id AND p.provider_code='volc-voice' AND p.deleted=0
WHERE pr.model_id='volc-tts-2-0' AND pr.price_version=1 AND pr.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price_detail` d
    WHERE d.price_id=pr.id AND d.token_type='input' AND d.time_slot='peak'
      AND d.min_tokens=0 AND d.max_tokens IS NULL AND d.deleted=0
  );
