-- ============================================================
-- 修正内置本地对话模型 qwen3-0.6b 的工具调用能力位
-- 模块: 核心模块-AI对话（模型管理 / 能力扩展）
-- ============================================================
-- 【OPT-IN：默认不执行，须人工执行】
--   问题：本地推理服务 clarity-llm 的 openai_compat 路由**不解析工具协议**（仅保证接口
--   形状兼容不报错，见 clarity-service/clarity-llm/app/routers/openai_compat.py），
--   故 qwen3-0.6b 无法产出原生 tool_calls；而 sys_ai_model 却把它标成
--   supports_tool_call=1（属于错误能力位），导致发送消息时按"支持工具"放行，
--   推理到工具阶段却永远不产生工具调用。
--   处置：置为 0（与事实一致），工具改走文本协议提示词注入（见
--   app/infrastructure/llm/tool_dialect.py）。种子 model_seeder.py 已同步修正，
--   本迁移用于修正既有环境（种子仅在模型行不存在时生效）。
-- 幂等：仅当 provider_code='local' 且 model_id='qwen3-0.6b' 且当前为 1 时更新。
-- 执行方式（不随 schema+data 自动导入）：
--   mysql -h<host> -u<user> -p<pass> <db> < 2026-10-02_fix_local_qwen_tool_call_capability.sql
-- 执行后须知：失效模型清单缓存 ai:model:list。
-- ============================================================
SET NAMES utf8mb4;

UPDATE `sys_ai_model` m
JOIN `sys_ai_provider` p ON p.id = m.provider_id AND p.provider_code = 'local' AND p.deleted = 0
SET m.`supports_tool_call` = 0,
    m.`update_time` = NOW()
WHERE m.`model_id` = 'qwen3-0.6b'
  AND m.`deleted` = 0
  AND m.`supports_tool_call` = 1;
