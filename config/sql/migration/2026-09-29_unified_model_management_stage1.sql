-- ============================================================
-- 2026-09-29 统一 AI 模型管理改造 · 阶段1（数据层）
-- 权威设计：clarity-doc/docs/03-模块设计/基础模块/AI模型管理/后端实现.md
-- 执行目标：clarity_dev（clarity_test 由 conftest/rebuild_mysql 按 schema+data 全量重建，不跑本文件）
-- ------------------------------------------------------------
-- 边界（纯增量）：
--   * 不 DROP 任何表、不删任何 schema 文件、不动菜单/权限种子、暂不删 voice_billing 字典键；
--   * 语音/算法数据并入统一表，但旧链路仍在跑（旧表保留，旧代码继续用旧表）。
-- 内容：
--   ① 四表列扩展 + 值域注释同步（sys_ai_model / _price / _price_detail / sys_ai_billing）
--   ② sys_ai_provider_key 唯一键放宽为 (provider_id, key_hash)
--   ③ 新增供应商 local-voice(8993/openai_audio)、local-modal(8994/openai_image) 及服务密钥
--   ④ sys_voice_model 3 行迁入 sys_ai_model（asr: sensevoice/stream+paraformer/offline；tts: huayan）
--   ⑤ sys_algorithm 有效行迁入 sys_ai_model（model_type=image_edit）
--   ⑥ dict 新增 ai_model.default_model.asr / .tts / .image_edit
-- 特性：幂等、可重复演练（加列/加索引均有存在性判断，插入均带 NOT EXISTS 守卫）。
-- 前置：建议三端后端停机执行；阶段1 仅增量 DDL（无删表），停服主要为避免搬迁瞬间的缓存口径不一致。
-- 回滚：见文件尾。
-- ============================================================
SET NAMES utf8mb4;

-- ---------- ① 四表列扩展（加列：MySQL 8.0 无 ADD COLUMN IF NOT EXISTS，用 information_schema 守卫） ----------

SET @sql := (SELECT IF(COUNT(*) = 0,
    'ALTER TABLE `sys_ai_model` ADD COLUMN `task_tags` json NULL DEFAULT NULL COMMENT ''任务归类标签(JSON数组;图像模型["clarity"]/["super_resolution"]…;asr模型["stream"]/["offline"];embedding模型["text"]/["image"])'' AFTER `model_type`',
    'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_ai_model' AND COLUMN_NAME = 'task_tags');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql := (SELECT IF(COUNT(*) = 0,
    'ALTER TABLE `sys_ai_model_price` ADD COLUMN `billing_unit` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT ''million_tokens'' COMMENT ''计费维度(million_tokens:百万token;image:张;second:秒;ten_thousand_chars:万字符;frame:帧)'' AFTER `unit`',
    'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_ai_model_price' AND COLUMN_NAME = 'billing_unit');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql := (SELECT IF(COUNT(*) = 0,
    'ALTER TABLE `sys_ai_model_price_detail` ADD COLUMN `usage_kind` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT ''token'' COMMENT ''计量粒度(token:按token;unit:按张/秒/万字符,与主表billing_unit联动)'' AFTER `token_type`',
    'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_ai_model_price_detail' AND COLUMN_NAME = 'usage_kind');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql := (SELECT IF(COUNT(*) = 0,
    'ALTER TABLE `sys_ai_billing` ADD COLUMN `quantity` decimal(18,4) NULL DEFAULT NULL COMMENT ''非token计量数量(如张数/秒数/万字符数;token类模型为NULL,见quantity_unit)'' AFTER `output_tokens`, ADD COLUMN `quantity_unit` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT ''非token计量单位(image:张;second:秒;ten_thousand_chars:万字符;frame:帧)'' AFTER `quantity`',
    'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_ai_billing' AND COLUMN_NAME = 'quantity');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 值域注释同步（model_type / protocol_type / token_type 扩展后，列注释须与白名单一致；MODIFY 可重跑）
ALTER TABLE `sys_ai_model`
    MODIFY COLUMN `model_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'chat' COMMENT '模型类型(chat:对话;embedding:向量;rerank:重排;image_gen:文生图;image_edit:图像编辑;video_gen:视频生成;asr:语音识别;tts:语音合成;music_gen:音乐生成;3d_gen:3D生成)';
ALTER TABLE `sys_ai_provider`
    MODIFY COLUMN `protocol_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'openai_compat' COMMENT '协议类型(openai_compat:OpenAI兼容;anthropic:Claude原生;openai_image:图像同步JSON;openai_audio:音频兼容;dashscope_async:百炼异步任务)';
ALTER TABLE `sys_ai_model_price_detail`
    MODIFY COLUMN `token_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '计费类型(input:输入未命中;cached:缓存命中;output:输出;unit:非token类单档)';

-- ---------- ② sys_ai_provider_key 唯一键放宽：全表 key_hash → (provider_id, key_hash) ----------
-- 背景：本地三供应商 local(8992)/local-voice(8993)/local-modal(8994) 共用同一把 INTERNAL_SERVICE_KEY，
--   全表唯一键会拒绝第 2/3 条登记（同 sys_voice_provider_key 2026-09-28 的同一处置）。
SET @sql := (SELECT IF(COUNT(*) >= 1,
    'ALTER TABLE `sys_ai_provider_key` DROP INDEX `uk_key_hash`, ADD UNIQUE INDEX `uk_provider_key` (`provider_id`, `key_hash`) USING BTREE',
    'SELECT 1')
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_ai_provider_key' AND INDEX_NAME = 'uk_key_hash');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------- ③ 新增本地供应商 local-voice(8993)/local-modal(8994) ----------
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 78,'local-voice','内置语音引擎','http://127.0.0.1:8993/v1','openai_audio','bearer',NULL,0,1,NULL,'内置本地语音服务（127.0.0.1:8993），Key 为服务密钥','1',0,2,2,NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_ai_provider` WHERE `provider_code`='local-voice' AND `deleted`=0);

INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT 79,'local-modal','内置图像引擎','http://127.0.0.1:8994/v1','openai_image','bearer',NULL,0,1,NULL,'内置本地图像服务（127.0.0.1:8994），Key 为服务密钥','1',0,2,2,NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_ai_provider` WHERE `provider_code`='local-modal' AND `deleted`=0);

-- 服务密钥迁入：从 sys_voice_provider_key（asr 引擎行，明文即 INTERNAL_SERVICE_KEY）复制密文到两个新供应商。
-- 密文由同一 aes_cipher（AI_PROVIDER_KEY_ENCRYPTION_KEY）加密，可跨表复制；key_hash 查重按 (provider_id, key_hash)。
INSERT INTO `sys_ai_provider_key` (`provider_id`,`name`,`key_hash`,`key_prefix`,`key_cipher`,`status`,`priority`,`weight`)
SELECT lv.id, vk.name, vk.key_hash, vk.key_prefix, vk.key_cipher, 1, vk.priority, vk.weight
FROM `sys_voice_provider_key` vk
JOIN `sys_voice_provider` vp ON vp.id = vk.provider_id AND vp.engine_type = 'asr'
JOIN `sys_ai_provider` lv ON lv.provider_code = 'local-voice' AND lv.deleted = 0
WHERE vk.status = 1
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_provider_key` k WHERE k.provider_id = lv.id AND k.key_hash = vk.key_hash);

INSERT INTO `sys_ai_provider_key` (`provider_id`,`name`,`key_hash`,`key_prefix`,`key_cipher`,`status`,`priority`,`weight`)
SELECT lm.id, vk.name, vk.key_hash, vk.key_prefix, vk.key_cipher, 1, vk.priority, vk.weight
FROM `sys_voice_provider_key` vk
JOIN `sys_voice_provider` vp ON vp.id = vk.provider_id AND vp.engine_type = 'asr'
JOIN `sys_ai_provider` lm ON lm.provider_code = 'local-modal' AND lm.deleted = 0
WHERE vk.status = 1
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_provider_key` k WHERE k.provider_id = lm.id AND k.key_hash = vk.key_hash);

-- ---------- ④ sys_voice_model 3 行迁入 sys_ai_model ----------
-- model_type：asr/tts；task_tags：asr 由 model_type 列（stream/offline）映射，tts 无 tag；
-- extra_request_params 不迁（onnx 路径/下载地址等运行时工件由 clarity-voice /v1/models 按 model_id 服务端解析，不落模型行）。
INSERT INTO `sys_ai_model` (`provider_id`,`model_id`,`model_type`,`task_tags`,`display_name`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT lv.id, vm.model_id, vm.engine_type,
       CASE WHEN vm.engine_type = 'asr' THEN JSON_ARRAY(vm.model_type) ELSE NULL END,
       vm.display_name, vm.status, 0, 0, 2, 2, NOW(), NOW()
FROM `sys_voice_model` vm
JOIN `sys_ai_provider` lv ON lv.provider_code = 'local-voice' AND lv.deleted = 0
WHERE vm.deleted = 0
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id = vm.model_id AND m.provider_id = lv.id AND m.deleted = 0);

-- ---------- ⑤-0 口径切换清理（一次性过渡，幂等） ----------
-- 本迁移初版曾以 model_id='algo-<id>' 产出行；定稿改为算法规范 slug（见第⑤段）。
-- 若目标库已有初版产出的行，必须先清理，否则滑入 slug 行后同一算法会出现两条模型
-- 且 dict 默认值仍指向 algo-1。此段在全新库/已切换库上均为 no-op。
-- 判定：仅 local-modal 下、model_type='image_edit'、未逻辑删除且 model_id 以 'algo-' 开头的行
--   （真实规范 slug 由 import_path/path 派生，不会以 'algo-' 开头）。
DELETE m FROM `sys_ai_model` m
JOIN `sys_ai_provider` p ON p.id = m.provider_id
WHERE p.provider_code = 'local-modal' AND p.deleted = 0
  AND m.model_type = 'image_edit' AND m.deleted = 0 AND m.model_id LIKE 'algo-%';

UPDATE `sys_dict`
SET `value` = 'aecrnet', `remark` = '平台默认图像编辑模型(须为已启用模型;aecrnet=AECR-Net)', `update_time` = NOW()
WHERE `type_code` = 'ai_model' AND `name` = 'default_model.image_edit' AND `value` LIKE 'algo-%';

-- ---------- ⑤ sys_algorithm 有效行迁入 sys_ai_model（image_edit） ----------
-- 只迁「可运行」行：path 非空（空 path 为分类/父节点，非模型），且 type<>'TEST'（测试残留）。
-- model_id 由「算法运行身份」派生（设计 §3.5/§4-2：model_id = 算法规范 slug，如 aecrnet）：
--   base   = norm(import_path 模块名)               -- algorithm.AECRNet.run → aecrnet
--   suffix = norm(path 去掉首段目录与扩展名)         -- C2PNet/ITS.pkl → its
--   同一 base 下存在多个不同 weights path 时取 base-suffix（c2pnet-its / c2pnet-ots），
--   否则取 base（aecrnet / aodnet）。norm = 小写、非 [a-z0-9] → '-'、折叠并去首尾 '-'。
--   实测：迁移 89 行 → 85 个唯一 model_id（最长 46 字符 < varchar(64)）；同 weights path 的重复行
--   （C2PNet/ITS.pkl id 3/4、DehazeFormer/reside6k/dehazeformer-b.pth id 38-41）合并保留 id 最小者：
--   同 provider 同权重即同一模型的多别名，display_name 可在管理端改，故不产生数据丢失（id 4/39/40/41 合并）。
--   完整 89 行映射表（sys_algorithm.id → model_id）见同目录
--   2026-09-29_unified_model_management_stage1_mapping.md。
-- status 6 态映射：已发布/测试中/待审核(4/2/3) → 1；草稿/停用/已归档(1/5/6) → 0。
-- task_tags 由 type 派生（仅映射实际出现的 4 类；未出现类型留空，不预留映射分支）。
INSERT INTO `sys_ai_model` (`provider_id`,`model_id`,`model_type`,`task_tags`,`display_name`,`status`,`vip_level`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT lm.id, q.model_id, 'image_edit', q.tag, q.display_name,
       CASE WHEN q.status IN (2, 3, 4) THEN 1 ELSE 0 END, 0, 0, 2, 2, NOW(), NOW()
FROM (
    SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.model_id ORDER BY t.id) AS rn
    FROM (
        SELECT s.id, s.display_name, s.status,
               CASE WHEN COUNT(*) OVER (PARTITION BY s.base) > 1
                    THEN CONCAT(s.base, '-', s.sfx) ELSE s.base END AS model_id,
               CASE s.type
                   WHEN '图像去雾' THEN JSON_ARRAY('clarity')
                   WHEN '图像去雨' THEN JSON_ARRAY('derain')
                   WHEN '图像去噪' THEN JSON_ARRAY('denoise')
                   WHEN '图像去模糊' THEN JSON_ARRAY('deblur')
                   ELSE NULL
               END AS tag
        FROM (
            SELECT a.id, a.name AS display_name, a.status, a.type,
                   TRIM(BOTH '-' FROM REGEXP_REPLACE(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(a.import_path, '^algorithm\\.', ''), '\\.run$', '')), '[^a-z0-9]+', '-')) AS base,
                   TRIM(BOTH '-' FROM REGEXP_REPLACE(LOWER(SUBSTRING_INDEX(IF(LOCATE('/', a.path) > 0, SUBSTRING(a.path, LOCATE('/', a.path) + 1), a.path), '.', 1)), '[^a-z0-9]+', '-')) AS sfx
            FROM `sys_algorithm` a
            WHERE a.deleted = 0 AND a.path IS NOT NULL AND a.path <> '' AND a.type <> 'TEST'
        ) s
    ) t
) q
JOIN `sys_ai_provider` lm ON lm.provider_code = 'local-modal' AND lm.deleted = 0
WHERE q.rn = 1
  AND NOT EXISTS (SELECT 1 FROM `sys_ai_model` m WHERE m.model_id = q.model_id AND m.provider_id = lm.id AND m.deleted = 0);

-- ---------- ⑤-1 迁入行状态修复（幂等） ----------
-- sys_ai_model.status 值域为 1:启用/0:禁用，而源表 sys_algorithm.status 是 6 态(1..6)；
-- 本迁移曾有一版把源值原样落库（出现 4/5）。插入守卫（NOT EXISTS）只判存在、不会更新已有行，
-- 故对已迁入行补一次修复。`status NOT IN (0,1)` 守卫保证幂等：第二次执行不会把已启用(1)的行打成禁用(0)。
UPDATE `sys_ai_model` m
JOIN `sys_ai_provider` p ON p.id = m.provider_id AND p.provider_code = 'local-modal' AND p.deleted = 0
SET m.status = CASE WHEN m.status IN (2, 3, 4) THEN 1 ELSE 0 END
WHERE m.model_type = 'image_edit' AND m.deleted = 0 AND m.status NOT IN (0, 1);

-- ---------- ⑥ dict 新增默认模型键（本阶段只播种，消费方在阶段 2/3 接入） ----------
INSERT INTO `sys_dict` (`type_code`,`name`,`value`,`sort`,`status`,`defaulted`,`remark`,`create_time`,`update_time`)
SELECT 'ai_model','default_model.asr','sensevoice',2,1,1,'平台默认语音识别模型(须为已启用模型)',NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_dict` WHERE `type_code`='ai_model' AND `name`='default_model.asr' AND `deleted`=0);

INSERT INTO `sys_dict` (`type_code`,`name`,`value`,`sort`,`status`,`defaulted`,`remark`,`create_time`,`update_time`)
SELECT 'ai_model','default_model.tts','huayan',3,1,1,'平台默认语音合成模型(须为已启用模型)',NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_dict` WHERE `type_code`='ai_model' AND `name`='default_model.tts' AND `deleted`=0);

INSERT INTO `sys_dict` (`type_code`,`name`,`value`,`sort`,`status`,`defaulted`,`remark`,`create_time`,`update_time`)
SELECT 'ai_model','default_model.image_edit','aecrnet',4,1,1,'平台默认图像编辑模型(须为已启用模型;aecrnet=AECR-Net)',NOW(),NOW()
FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM `sys_dict` WHERE `type_code`='ai_model' AND `name`='default_model.image_edit' AND `deleted`=0);

-- ============================================================
-- 回滚（按需，逆序）：
--   DELETE FROM sys_dict WHERE type_code='ai_model' AND name IN ('default_model.asr','default_model.tts','default_model.image_edit');
--   DELETE m FROM sys_ai_model m JOIN sys_ai_provider p ON p.id=m.provider_id WHERE m.model_type IN ('image_edit','asr','tts') AND p.provider_code IN ('local-voice','local-modal');
--   DELETE k FROM sys_ai_provider_key k JOIN sys_ai_provider p ON p.id=k.provider_id WHERE p.provider_code IN ('local-voice','local-modal');
--   DELETE FROM sys_ai_provider WHERE provider_code IN ('local-voice','local-modal');
--   ALTER TABLE sys_ai_provider_key DROP INDEX uk_provider_key, ADD UNIQUE INDEX uk_key_hash (key_hash);
--   ALTER TABLE sys_ai_billing DROP COLUMN quantity_unit, DROP COLUMN quantity;
--   ALTER TABLE sys_ai_model_price_detail DROP COLUMN usage_kind;
--   ALTER TABLE sys_ai_model_price DROP COLUMN billing_unit;
--   ALTER TABLE sys_ai_model DROP COLUMN task_tags;
-- （列注释 MODIFY 不回滚：注释为元数据，无功能影响）
-- ============================================================
