-- ============================================================
-- 平台默认模型切换：切到第三方（硅基流动 chat + 火山方舟/火山语音）
-- 模块: 基础模块-AI模型管理
-- ============================================================
-- 【OPT-IN：默认不执行，须在密钥就位后手工执行】
--   前置：已在管理端为 volc-ark / volc-voice /
--         siliconflow 写入密钥（sys_ai_provider_key，AES 加密入库），否则 chat/asr/tts 会直接失败。
--   本文件是"登记供应商与模型"（config/sql/data/ 种子）之后的**独立切换步骤**：
--   种子只登记 volc-ark / volc-voice / siliconflow 与对应模型并保持默认模型仍指向
--   当前可用的内置本地模型（qwen3-0.6b / sensevoice / huayan），避免无密钥时打断链路。
--   默认 chat 取硅基流动 Qwen/Qwen3-8B（免费档）；方舟 doubao-seed-2-0-mini-260428 作为同能力
--   备选（按 token 单价更低但依赖方舟免费额度），如需改用把下面的取值替换即可。
--   执行方式（不随 schema+data 自动导入，schema/gen 不会跑本文件）：
--     mysql -h<host> -u<user> -p<pass> <db> < 2026-10-01_switch_default_model_volc.sql
--   幂等：仅按 name 更新已存在字典行的取值，重复执行结果一致、不产生重复行。
--   执行后需失效缓存（默认模型与模型/供应商清单）：ai:model:list、ai:provider:list、ai:rate:{model_id}，
--   以及**字典值缓存 `dict:value:ai_model:*`**（get_dict_value 带 1h 缓存；本文件用原生 SQL 改字典、
--   绕过服务失效路径，不清该前缀会出现"库里已切、服务仍解析旧默认模型"——2026-10-02 实测）。
--     （标准入口：dehaze-test/scripts/rebuild_mysql.py --skip-mysql）
-- ============================================================
SET NAMES utf8mb4;

-- 默认 chat：硅基流动 Qwen/Qwen3-8B（免费档，0 现金成本；16 元券亦可用于其它模型）
UPDATE `sys_dict` SET `value` = 'Qwen/Qwen3-8B', `update_time` = NOW()
WHERE `type_code` = 'ai_model' AND `name` = 'default_model' AND `deleted` = 0;

UPDATE `sys_dict` SET `value` = 'bigmodel', `update_time` = NOW()
WHERE `type_code` = 'ai_model' AND `name` = 'default_model.asr' AND `deleted` = 0;

UPDATE `sys_dict` SET `value` = 'volc-tts-2-0', `update_time` = NOW()
WHERE `type_code` = 'ai_model' AND `name` = 'default_model.tts' AND `deleted` = 0;
