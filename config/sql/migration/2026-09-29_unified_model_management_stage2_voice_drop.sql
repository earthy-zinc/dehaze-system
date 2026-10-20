-- ============================================================
-- 2026-09-29 统一 AI 模型管理改造 · 阶段2（语音旧表 DROP）
-- 权威设计：clarity-doc/docs/03-模块设计/基础模块/AI模型管理/后端实现.md
-- 执行方式：mysql --default-character-set=utf8mb4 <本文件>（需 mysql CLI，过程体依赖 DELIMITER）
-- 执行目标：clarity_dev（clarity_test 由 conftest 按 schema+data 全量重建，不跑本文件）
-- ------------------------------------------------------------
-- 前置（全部满足才允许 DROP，设计 §6.1 不可回退点保护）：
--   1. python 语音链路已切统一路由并验证（阶段 2 代码合入）；
--   2. sys_ai_model 已含 asr/tts 模型、local-voice 供应商、密钥与价格（阶段 1 已落地）；
--   3. sys_voice_model 每个未删除行都在统一表有对应（缺失即中止）。
-- 幂等：voice 三表已不存在时校验段整体跳过，DROP 段 no-op。
-- 回滚：不可回退（DROP 后无 DDL 可恢复）；执行前 mysqldump 备份三表与 sys_dict 相关行。
-- ============================================================
SET NAMES utf8mb4;

DROP PROCEDURE IF EXISTS `_stage2_voice_drop`;

DELIMITER $$
CREATE PROCEDURE `_stage2_voice_drop`()
BEGIN
    DECLARE v_exist INT DEFAULT 0;
    DECLARE v_msg TEXT DEFAULT NULL;

    SELECT COUNT(*) INTO v_exist FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME IN ('sys_voice_provider', 'sys_voice_provider_key', 'sys_voice_model');

    IF v_exist = 0 THEN
        -- 已切换完成，整体 no-op
        SELECT 'voice 三表已不存在，本迁移为 no-op' AS result;
    ELSE
        -- ① 统一表必须有 asr/tts 模型行（阶段 1 迁入 sensevoice/paraformer/huayan）
        IF v_msg IS NULL AND (SELECT COUNT(*) FROM `sys_ai_model` WHERE model_type IN ('asr', 'tts') AND deleted = 0) < 3 THEN
            SET v_msg = '统一表 asr/tts 模型行不足 3，阶段 1 迁移未完成，拒绝 DROP';
        END IF;

        -- ② local-voice 供应商必须存在且启用
        IF v_msg IS NULL AND (SELECT COUNT(*) FROM `sys_ai_provider`
            WHERE provider_code = 'local-voice' AND deleted = 0 AND status = 1) = 0 THEN
            SET v_msg = 'sys_ai_provider 缺少启用的 local-voice，拒绝 DROP';
        END IF;

        -- ③ local-voice 必须有启用密钥（sys_ai_provider_key 无逻辑删除列，仅按 status 判定）
        IF v_msg IS NULL AND (SELECT COUNT(*) FROM `sys_ai_provider_key` k
            JOIN `sys_ai_provider` p ON p.id = k.provider_id
            WHERE p.provider_code = 'local-voice' AND k.status = 1) = 0 THEN
            SET v_msg = 'local-voice 无启用密钥，拒绝 DROP';
        END IF;

        -- ④ asr/tts 模型必须有有效价格版本（阶段 1 seeder 已落 second/ten_thousand_chars）
        IF v_msg IS NULL AND (SELECT COUNT(DISTINCT p.model_id) FROM `sys_ai_model_price` p
            JOIN `sys_ai_model` m ON m.model_id = p.model_id
            WHERE m.model_type IN ('asr', 'tts') AND p.status = 1 AND p.deleted = 0
              AND p.effective_from <= NOW()
              AND (p.effective_to IS NULL OR p.effective_to > NOW())) < 3 THEN
            SET v_msg = 'asr/tts 模型缺有效价格版本，拒绝 DROP';
        END IF;

        -- ⑤ 数据迁出校验：sys_voice_model 每个未删除行在统一表有对应
        IF v_msg IS NULL AND (SELECT COUNT(*) FROM `sys_voice_model` vm
            WHERE vm.deleted = 0
              AND NOT EXISTS (
                  SELECT 1 FROM `sys_ai_model` m
                  JOIN `sys_ai_provider` p ON p.id = m.provider_id AND p.provider_code = 'local-voice'
                  WHERE m.model_id = vm.model_id AND m.model_type = vm.engine_type AND m.deleted = 0
              )) > 0 THEN
            SET v_msg = 'sys_voice_model 存在未迁入统一表的行，先补迁移再 DROP';
        END IF;

        IF v_msg IS NOT NULL THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = v_msg;
        END IF;

        DROP TABLE IF EXISTS `sys_voice_provider`;
        DROP TABLE IF EXISTS `sys_voice_provider_key`;
        DROP TABLE IF EXISTS `sys_voice_model`;

        SELECT 'voice 三表已 DROP（阶段 2 不可回退点已执行）' AS result;
    END IF;
END$$
DELIMITER ;

CALL `_stage2_voice_drop`();
DROP PROCEDURE IF EXISTS `_stage2_voice_drop`;

-- ---------- dict：voice_billing 单价键废除（计费已迁模型价格表） ----------
DELETE FROM `sys_dict`
WHERE `type_code` = 'voice_billing'
  AND `name` IN ('asr_credits_per_second', 'tts_credits_per_char')
  AND `deleted` = 0;

-- sys_voice_hotword 保留：热词属用户端 /api/v1/voice/* 域（设计 D7），不随引擎体系退役。
