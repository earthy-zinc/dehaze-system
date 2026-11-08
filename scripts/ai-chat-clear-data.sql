-- ============================================================
-- AI 对话域存量数据清空（一次性运维动作，不可回滚）
-- 依据: clarity-doc/docs/03-模块设计/核心模块/AI对话/架构收敛方案.md
-- ============================================================
-- ⚠️ 执行前必须确认目标库（本脚本不校验环境，连错库即不可逆）：
--    SELECT DATABASE() AS target_database, @@hostname AS target_host;
--    核对返回值与计划清理的环境（dev/pre/prod）一致后再继续。
--
-- 背景：AI 对话 v2 契约不兼容旧数据，切换前清空对话域，由新实现从零写入；
--      代码里不保留任何兼容旧数据的分支。
--
-- 清空范围（仅 AI 对话域，共 5 张表）：
--   sys_ai_conversation    会话
--   sys_ai_message         消息（v2 结构，旧消息一律丢弃）
--   sys_ai_agent_thought   推理步骤
--   sys_ai_artifact        中间产物引用
--   sys_ai_message_feedback AI 消息反馈
--
-- 明确不动（账务与可观测流水，涉及对账，禁止随对话域清理）：
--   sys_ai_billing / sys_ai_credit_log / sys_ai_trace / sys_ai_llm_call
--
-- 脚本之外还需人工清理：ES 中对应的消息索引与 trace 记录（本文件只处理 MySQL）。
-- 执行顺序：先停旧版写入 → 本脚本 → 迁移 DDL（config/sql/migration/2026-10-21_ai_chat_v2_destructive.sql）
--           → 清理 ES → 切换流量。
-- ------------------------------------------------------------

-- 环境确认（只读；执行 TRUNCATE 前先看这一行的输出）
SELECT DATABASE() AS target_database, @@hostname AS target_host;

-- 子表先清，父表后清（当前 schema 未定义外键，顺序仍按依赖排列，便于人工核对）
TRUNCATE TABLE `sys_ai_artifact`;
TRUNCATE TABLE `sys_ai_agent_thought`;
TRUNCATE TABLE `sys_ai_message_feedback`;
TRUNCATE TABLE `sys_ai_message`;
TRUNCATE TABLE `sys_ai_conversation`;

-- 清理结果核对：五张表应全部为 0
SELECT 'sys_ai_conversation' AS table_name, COUNT(*) AS row_count FROM `sys_ai_conversation`
UNION ALL SELECT 'sys_ai_message', COUNT(*) FROM `sys_ai_message`
UNION ALL SELECT 'sys_ai_agent_thought', COUNT(*) FROM `sys_ai_agent_thought`
UNION ALL SELECT 'sys_ai_artifact', COUNT(*) FROM `sys_ai_artifact`
UNION ALL SELECT 'sys_ai_message_feedback', COUNT(*) FROM `sys_ai_message_feedback`;
