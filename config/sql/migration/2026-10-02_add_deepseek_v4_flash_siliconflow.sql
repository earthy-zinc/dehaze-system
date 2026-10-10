-- ============================================================
-- 登记更强模型：deepseek-ai/DeepSeek-V4-Flash（硅基流动）
-- 模块: 核心模块-AI对话（模型管理 / Agent 评测）
-- ============================================================
-- 【OPT-IN：默认不执行，须人工执行】
--   背景：平台默认 chat 模型为 Qwen/Qwen3-8B（8B 小模型），在需要较高模型智力的场景
--   （Agent 评测 F-M08-014 的工具调用准确率/步骤正确性、plan_execute/reflexion 多步推理、
--   工具密集任务）能力不足。硅基流动已托管 deepseek-ai/DeepSeek-V4-Flash 且现有密钥可用
--   ⇒ 登记为可选 chat 模型（不做默认；默认仍 Qwen3-8B 控成本），供高智力测试/评测选用。
--   （另有 deepseek-ai/DeepSeek-V4-Pro 更高档，如需可另登记。）
-- 真实可用性：登记后可在管理端用「模型可用性测试」验证；provider = siliconflow（已配密钥）。
-- 定价：暂按 DeepSeek 官方公开价折算（口径 1 元 = 100 积分）：
--   输入(缓存未命中) ¥1/M、输出 ¥2/M、缓存命中 ¥0.02/M
--   ⇒ input 100 / output 200 / cached 2 积分/百万token。
--   ⚠️ 硅基流动实际结算价可能不同：上线前请在管理端核对并按需新增价格版本（price_version+1）。
-- 幂等：模型行按 (model_id, provider_id) 守卫；价格按 (model_id, provider_id, price_version) 与档位守卫。
-- 执行方式（不随 schema+data 自动导入）：
--   mysql -h<host> -u<user> -p<pass> <db> < 2026-10-02_add_deepseek_v4_flash_siliconflow.sql
-- 执行后须知：失效模型/供应商清单缓存 ai:model:list、ai:provider:list。
-- ============================================================
SET NAMES utf8mb4;

-- 1) 登记模型
INSERT INTO `sys_ai_model`
  (`provider_id`,`model_id`,`model_type`,`task_tags`,`dimension`,`display_name`,`max_context_tokens`,`max_output_tokens`,`supports_multimodal`,`supports_tool_call`,`supports_streaming`,`supports_prompt_cache`,`supports_structured_output`,`extra_request_params`,`fallback_model_id`,`prompt_cache_prefix_len`,`image_tokens_per_image`,`concurrency_limit`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT p.id,'deepseek-ai/DeepSeek-V4-Flash','chat',NULL,NULL,'DeepSeek V4 Flash（硅基流动）',128000,8192,0,1,1,1,0,NULL,NULL,0,0,NULL,1,0,0,2,2,NOW(),NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='siliconflow' AND p.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model` m
    WHERE m.model_id='deepseek-ai/DeepSeek-V4-Flash' AND m.provider_id=p.id
  );

-- 2) 价格版本 v1
INSERT INTO `sys_ai_model_price`
  (`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 'deepseek-ai/DeepSeek-V4-Flash', p.id, 1, 'credits_per_million', 'million_tokens', '2026-10-02 00:00:00', NULL, 1, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code='siliconflow' AND p.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price` pr
    WHERE pr.model_id='deepseek-ai/DeepSeek-V4-Flash' AND pr.provider_id=p.id AND pr.price_version=1
  );

-- 3) 价格档位明细（单档 0..NULL，峰谷同价）
INSERT INTO `sys_ai_model_price_detail`
  (`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT pr.id, t.token_type, 'token', t.time_slot, 0, NULL, t.unit_price, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_model_price` pr
JOIN `sys_ai_provider` p ON p.id = pr.provider_id AND p.provider_code='siliconflow' AND p.deleted=0
JOIN (
            SELECT 'input'  AS token_type, 'peak' AS time_slot, 100.0000 AS unit_price
  UNION ALL SELECT 'input',  'idle', 100.0000
  UNION ALL SELECT 'cached', 'peak', 2.0000
  UNION ALL SELECT 'cached', 'idle', 2.0000
  UNION ALL SELECT 'output', 'peak', 200.0000
  UNION ALL SELECT 'output', 'idle', 200.0000
) t
WHERE pr.model_id='deepseek-ai/DeepSeek-V4-Flash' AND pr.price_version=1 AND pr.deleted=0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price_detail` d
    WHERE d.price_id=pr.id AND d.token_type=t.token_type AND d.time_slot=t.time_slot
      AND d.min_tokens=0 AND d.max_tokens IS NULL AND d.deleted=0
  );
