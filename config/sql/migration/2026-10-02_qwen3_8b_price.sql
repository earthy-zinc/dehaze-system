-- ============================================================
-- Qwen/Qwen3-8B（硅基流动）登记非免费用户售价
-- 模块: 核心模块-AI对话（计费）
-- ============================================================
-- 【OPT-IN：默认不执行，须人工执行，与默认模型切换配套】
--   背景：默认模型已切 Qwen/Qwen3-8B（见 2026-10-01_switch_default_model_volc.sql），
--   但该模型未登记用户售价 ⇒ 结算 rate_provider.calculate 抛「模型 Qwen/Qwen3-8B 未配置用户售价」，
--   每次调用 settle 失败（预扣已落库、待对账），积分不落库。
-- 定价口径：兑换锚 1 元 = 100 积分（见 03-模块设计/基础模块/套餐管理/需求规格.md）。
--   硅基流动 Qwen/Qwen3-8B 公开价约 输入 ¥0.35/百万token、输出 ¥1.40/百万token
--   ⇒ input 35 / output 140 积分/百万token；该模型 supports_prompt_cache=0，cached 与 input 同价。
--   非免费（明确不设 0 价）。不分上下文分段（模型 max_context_tokens=32768，单档 0..NULL）。
-- 执行后须知：失效费率缓存 ai:rate:Qwen/Qwen3-8B（TTL 30 分钟，或主动删该键）。
-- 幂等：(model_id, provider_id, price_version) 与档位 (price_id, token_type, time_slot, 分段) 均做存在性守卫。
-- 执行方式（不随 schema+data 自动导入）：
--   mysql -h<host> -u<user> -p<pass> <db> < 2026-10-02_qwen3_8b_price.sql
-- ============================================================
SET NAMES utf8mb4;

-- 去掉「免费档」字样，避免与实际收费不符
UPDATE `sys_ai_model` m
JOIN `sys_ai_provider` p ON p.id = m.provider_id AND p.provider_code = 'siliconflow' AND p.deleted = 0
SET m.display_name = 'Qwen3-8B（硅基流动）', m.update_time = NOW()
WHERE m.model_id = 'Qwen/Qwen3-8B' AND m.deleted = 0;

-- 价格版本 v1
INSERT INTO `sys_ai_model_price`
  (`model_id`,`provider_id`,`price_version`,`unit`,`billing_unit`,`effective_from`,`effective_to`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 'Qwen/Qwen3-8B', p.id, 1, 'credits_per_million', 'million_tokens', '2026-10-01 00:00:00', NULL, 1, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_provider` p
WHERE p.provider_code = 'siliconflow' AND p.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price` pr
    WHERE pr.model_id = 'Qwen/Qwen3-8B' AND pr.provider_id = p.id AND pr.price_version = 1
  );

-- 档位明细：三类型 × 峰谷同价，单档 min_tokens=0 / max_tokens=NULL
INSERT INTO `sys_ai_model_price_detail`
  (`price_id`,`token_type`,`usage_kind`,`time_slot`,`min_tokens`,`max_tokens`,`unit_price`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT pr.id, t.token_type, 'token', t.time_slot, 0, NULL, t.unit_price, 0, 2, 2, NOW(), NOW()
FROM `sys_ai_model_price` pr
JOIN `sys_ai_provider` p ON p.id = pr.provider_id AND p.provider_code = 'siliconflow' AND p.deleted = 0
JOIN (
            SELECT 'input'  AS token_type, 'peak' AS time_slot, 35.0000  AS unit_price
  UNION ALL SELECT 'input',  'idle', 35.0000
  UNION ALL SELECT 'cached', 'peak', 35.0000
  UNION ALL SELECT 'cached', 'idle', 35.0000
  UNION ALL SELECT 'output', 'peak', 140.0000
  UNION ALL SELECT 'output', 'idle', 140.0000
) t
WHERE pr.model_id = 'Qwen/Qwen3-8B' AND pr.price_version = 1 AND pr.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_model_price_detail` d
    WHERE d.price_id = pr.id AND d.token_type = t.token_type AND d.time_slot = t.time_slot
      AND d.min_tokens = 0 AND d.max_tokens IS NULL AND d.deleted = 0
  );
