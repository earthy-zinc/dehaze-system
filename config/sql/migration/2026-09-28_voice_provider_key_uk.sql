-- ============================================================
-- 语音引擎 API Key 唯一键：全局 key_hash → (provider_id, key_hash)
-- 背景：语音引擎按能力注册多条供应商行（内置引擎 asr/tts 两条共用同一把
-- INTERNAL_SERVICE_KEY；某厂商 asr/tts 共用一把 AccessKey），原全局唯一键
-- uk_key_hash(key_hash) 会拒绝第二条登记 —— 实测后果：内置引擎 tts 播种缺密钥
-- （IntegrityError/查重跳过），调用不带 Authorization，引擎侧 401。
-- 注意：一次性执行，不可重复执行（DROP INDEX 无 IF EXISTS，重跑会报错）。
-- 执行顺序：先确认无重复 (provider_id, key_hash)（本表语音密钥功能未上线，
--   目标库预期 0 行）再执行；存量若有冲突需先人工处置。
-- ============================================================

ALTER TABLE `sys_voice_provider_key`
    DROP INDEX `uk_key_hash`,
    ADD UNIQUE INDEX `uk_provider_key` (`provider_id`, `key_hash`) USING BTREE;
