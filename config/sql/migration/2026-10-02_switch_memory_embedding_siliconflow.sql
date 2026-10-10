-- ============================================================
-- 记忆 embedding 切换到硅基流动 BGE-M3
-- 模块: 核心模块-AI对话（长期记忆）
-- ============================================================
-- 【OPT-IN：默认不执行，须人工执行】
--   背景：sys_dict.ai_embedding 原指向 openai / text-embedding-3-small，而 openai 供应商
--   无可用 Key ⇒ 记忆向量同步恒失败（Embedding 调用失败…无可用 API Key）、ES 记忆索引恒空，
--   记忆语义检索失效、且无向量时仅能整句关键词兜底（几乎召不回）。
--   改用已配置密钥的硅基流动 BAAI/bge-m3（dimension=1024，登记见 2026-10-01 种子迁移）。
-- 执行后须知：
--   1) ES 记忆索引维度由 1536 变 1024：旧索引（{ES_INDEX_PREFIX}ai_memory）若为空可直接删除，
--      重启 Python 服务时 ensure_memory_index() 会按新 dims 重建；若已有文档须先重建再回灌。
--   2) 已有 MySQL 记忆需一次性回灌向量（逐条 sync_memory），否则旧记忆检索取不到。
--   3) 失效字典缓存 dict:value:ai_embedding:*（get_dict_value 带 1h 缓存）。
--   4) 三端后端需重启（embedding 配置在进程内按 dict 解析）。
-- 幂等：仅按 (type_code,name) 更新已存在字典行的取值，重复执行结果一致、不产生重复行。
-- 执行方式（不随 schema+data 自动导入）：
--   mysql -h<host> -u<user> -p<pass> <db> < 2026-10-02_switch_memory_embedding_siliconflow.sql
-- ============================================================
SET NAMES utf8mb4;

UPDATE `sys_dict` SET `value` = 'siliconflow', `update_time` = NOW()
WHERE `type_code` = 'ai_embedding' AND `name` = 'provider_code' AND `deleted` = 0;

UPDATE `sys_dict` SET `value` = 'BAAI/bge-m3', `update_time` = NOW()
WHERE `type_code` = 'ai_embedding' AND `name` = 'model' AND `deleted` = 0;

UPDATE `sys_dict` SET `value` = '1024', `update_time` = NOW()
WHERE `type_code` = 'ai_embedding' AND `name` = 'dims' AND `deleted` = 0;
