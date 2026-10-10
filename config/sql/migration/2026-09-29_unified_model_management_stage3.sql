-- ============================================================
-- 2026-09-29 统一 AI 模型管理改造 · 阶段3（图像链路模型化 + 算法两表 DROP）
-- 权威设计：dehaze-doc/docs/03-模块设计/基础模块/AI模型管理/后端实现.md
-- 执行方式：mysql --default-character-set=utf8mb4 <本文件>（过程体依赖 DELIMITER）
-- 执行目标：dehaze_dev（dehaze_test 由 conftest 按 schema+data 全量重建，不跑本文件）
-- ------------------------------------------------------------
-- 前置（全部满足才允许执行到 DROP 段）：
--   1. python / java / go 三端图像链路已切模型化并验证（阶段 3 代码合入）；
--   2. sys_ai_model 已含 image_edit 模型、启用的 local-modal 供应商与价格（阶段 1 已落地）；
--   3. sys_algorithm 每个未删除行的权重路径已随阶段 1 迁入统一模型表。
-- 幂等：sys_algorithm 已不存在时整体 no-op；各表按旧列是否存在逐表跳过。
-- 回滚：DROP 段不可回退；执行前 mysqldump 备份 sys_algorithm(_version) 与
--       sys_pred_log / sys_eval_log / sys_rating / sys_recommendation(_rule) /
--       sys_input_history / sys_preset / sys_favorite / sys_member_benefit。
-- 说明：sys_member.monthly_* 与 sys_member_benefit.monthly_*_quota 列**保留不删**（设计 §9.1 第 3 条），
--       仅停止读写；本文件把 benefit 的配额列置 0。
-- ============================================================
SET NAMES utf8mb4;

-- ---------- 映射表：sys_algorithm.id → 统一模型规范 slug（阶段 1 迁移同规则，含被合并行；
--            另补历史默认算法 DCP(id=13, 无权重路径未迁入) → 平台默认模型 aecrnet） ----------
DROP TEMPORARY TABLE IF EXISTS `_algo_slug`;
CREATE TEMPORARY TABLE `_algo_slug`
(
    `algo_id` int         NOT NULL,
    `slug`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (`algo_id`)
);
INSERT INTO `_algo_slug` (`algo_id`, `slug`)
VALUES
  (13, 'aecrnet'),
  (1, 'aecrnet'),
  (2, 'aodnet'),
  (3, 'c2pnet-its'),
  (4, 'c2pnet-its'),
  (5, 'c2pnet-ots'),
  (7, 'cmfnet-dehaze-i-ohaze-cmfnet'),
  (8, 'cmfnet-deblur-gopro-cmfnet'),
  (9, 'cmfnet-deraindrop-deraindrop-cmfnet'),
  (10, 'd4'),
  (11, 'daclipuir'),
  (12, 'dcpdn'),
  (15, 'deanet-haze4k-psnr3426-ssim9885'),
  (16, 'deanet-its-psnr4131-ssim9945'),
  (17, 'deanet-ots-psnr3659-ssim9897'),
  (19, 'dehamer-dense-psnr1662-ssim05602'),
  (20, 'dehamer-indoor-psnr3663-ssim09881'),
  (21, 'dehamer-nh-psnr2066-ssim06844'),
  (22, 'dehamer-outdoor-psnr3518-ssim09860'),
  (25, 'dehazeformer-indoor-dehazeformer-b'),
  (26, 'dehazeformer-indoor-dehazeformer-d'),
  (27, 'dehazeformer-indoor-dehazeformer-l'),
  (28, 'dehazeformer-indoor-dehazeformer-m'),
  (29, 'dehazeformer-indoor-dehazeformer-s'),
  (30, 'dehazeformer-indoor-dehazeformer-t'),
  (31, 'dehazeformer-indoor-dehazeformer-w'),
  (33, 'dehazeformer-outdoor-dehazeformer-b'),
  (34, 'dehazeformer-outdoor-dehazeformer-m'),
  (35, 'dehazeformer-outdoor-dehazeformer-s'),
  (36, 'dehazeformer-outdoor-dehazeformer-t'),
  (38, 'dehazeformer-reside6k-dehazeformer-b'),
  (39, 'dehazeformer-reside6k-dehazeformer-b'),
  (40, 'dehazeformer-reside6k-dehazeformer-b'),
  (41, 'dehazeformer-reside6k-dehazeformer-b'),
  (43, 'dehazeformer-rshaze-dehazeformer-b'),
  (44, 'dehazeformer-rshaze-dehazeformer-m'),
  (45, 'dehazeformer-rshaze-dehazeformer-s'),
  (46, 'dehazeformer-rshaze-dehazeformer-t'),
  (47, 'fcd'),
  (49, 'ffanet-its-train-ffa-3-19'),
  (50, 'ffanet-ots-train-ffa-3-19'),
  (51, 'fogremoval'),
  (52, 'gcanet'),
  (54, 'griddehazenet-indoor-haze-best-3-6'),
  (55, 'griddehazenet-outdoor-haze-best-3-6'),
  (57, 'imgrestorationsde-deblurring-ir-sde-deblurring'),
  (58, 'imgrestorationsde-denoising-ir-sde-denoising'),
  (60, 'imgrestorationsde-deraining-ir-sde-derainh100'),
  (61, 'imgrestorationsde-deraining-ir-sde-derainl100'),
  (62, 'itbdehaze'),
  (63, 'lightdehazenet'),
  (66, 'lkdnet-its-lkd-b-lkd-b'),
  (67, 'lkdnet-its-lkd-l-lkd-l'),
  (68, 'lkdnet-its-lkd-s-lkd-s'),
  (69, 'lkdnet-its-lkd-t-lkd-t'),
  (71, 'lkdnet-ots-lkd-b-lkd-b'),
  (72, 'lkdnet-ots-lkd-l-lkd-l'),
  (73, 'lkdnet-ots-lkd-s-lkd-s'),
  (74, 'lkdnet-ots-lkd-t-lkd-t'),
  (75, 'madn'),
  (78, 'mb-taylorformer-densehaze-mb-taylorformer-b'),
  (79, 'mb-taylorformer-densehaze-mb-taylorformer-l'),
  (80, 'mb-taylorformer-its-mb-taylorformer-l'),
  (81, 'mb-taylorformer-ohaze-mb-taylorformer-b'),
  (83, 'mb-taylorformer-ots-mb-taylorformer-b'),
  (84, 'mb-taylorformer-ots-mb-taylorformer-l'),
  (87, 'mixdehazenet-haze4k-mixdehazenet-l'),
  (89, 'mixdehazenet-indoor-mixdehazenet-b'),
  (90, 'mixdehazenet-indoor-mixdehazenet-l'),
  (92, 'mixdehazenet-outdoor-mixdehazenet-b'),
  (93, 'mixdehazenet-outdoor-mixdehazenet-l'),
  (94, 'mixdehazenet-outdoor-mixdehazenet-s'),
  (96, 'msfnet-indoor'),
  (97, 'msfnet-outdoor'),
  (99, 'psd-psb-msbdn'),
  (100, 'psd-psd-ffanet'),
  (101, 'psd-psd-gcanet'),
  (102, 'ridcp'),
  (104, 'scanet-gmodel-40'),
  (105, 'scanet-gmodel-105'),
  (106, 'scanet-gmodel-120'),
  (108, 'sgidpff-sots-indoor'),
  (109, 'sgidpff-sots-outdoor'),
  (110, 'tsdnet'),
  (112, 'wpxnet-dense-haze'),
  (113, 'wpxnet-i-haze'),
  (114, 'wpxnet-o-haze'),
  (115, 'wpxnet-nh-haze-20'),
  (116, 'wpxnet-nh-haze-21'),
  (117, 'wpxnet-nh-haze-23');

DROP TEMPORARY TABLE IF EXISTS `_rec_models`;
DROP PROCEDURE IF EXISTS `_stage3_image_migrate`;

DELIMITER $$
CREATE PROCEDURE `_stage3_image_migrate`()
BEGIN
    DECLARE v_exists INT DEFAULT 0;
    DECLARE v_msg TEXT DEFAULT NULL;

    SELECT COUNT(*) INTO v_exists FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sys_algorithm';

    IF v_exists = 0 THEN
        SELECT 'sys_algorithm 已不存在，阶段 3 迁移为 no-op' AS result;
    ELSE
        -- ① 统一表 image_edit 模型与启用的 local-modal 供应商必须就绪
        IF (SELECT COUNT(*) FROM `sys_ai_model` WHERE model_type = 'image_edit' AND deleted = 0) < 85 THEN
            SET v_msg = '统一表 image_edit 模型行不足 85，阶段 1 迁移未完成，拒绝执行';
        END IF;
        IF v_msg IS NULL AND (SELECT COUNT(*) FROM `sys_ai_provider`
                WHERE provider_code = 'local-modal' AND deleted = 0 AND status = 1) = 0 THEN
            SET v_msg = 'sys_ai_provider 缺少启用的 local-modal，拒绝执行';
        END IF;
        IF v_msg IS NOT NULL THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = v_msg;
        END IF;

        -- ================= sys_pred_log =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_pred_log' AND COLUMN_NAME = 'algorithm_id') > 0 THEN
            ALTER TABLE `sys_pred_log`
                ADD COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL AFTER `id`,
                ADD COLUMN `provider_id` bigint NULL DEFAULT NULL COMMENT '实际调用供应商id（归因）' AFTER `model_id`,
                ADD COLUMN `actual_model` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '实际执行模型' AFTER `provider_id`,
                ADD COLUMN `vendor_task_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '异步厂商任务号（本地同步推理为空）' AFTER `actual_model`;
            UPDATE `sys_pred_log` p JOIN `_algo_slug` s ON s.algo_id = p.algorithm_id SET p.model_id = s.slug;
            UPDATE `sys_pred_log` SET `model_id` = CONCAT('legacy-', `algorithm_id`) WHERE `model_id` IS NULL;
            UPDATE `sys_pred_log` SET `actual_model` = `model_id` WHERE `actual_model` IS NULL;
            ALTER TABLE `sys_pred_log` MODIFY COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图像模型标识（统一模型规范 slug）';
            IF (SELECT COUNT(*) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'sys_pred_log' AND INDEX_NAME = 'idx_algorithm_id') > 0 THEN
                ALTER TABLE `sys_pred_log` DROP INDEX `idx_algorithm_id`;
            END IF;
            ALTER TABLE `sys_pred_log` DROP COLUMN `algorithm_id`;
            ALTER TABLE `sys_pred_log` ADD INDEX `idx_model_id` (`model_id`) USING BTREE;
        END IF;

        -- ================= sys_eval_log =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_eval_log' AND COLUMN_NAME = 'algorithm_id') > 0 THEN
            ALTER TABLE `sys_eval_log`
                ADD COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL AFTER `id`;
            UPDATE `sys_eval_log` e JOIN `_algo_slug` s ON s.algo_id = e.algorithm_id SET e.model_id = s.slug;
            UPDATE `sys_eval_log` SET `model_id` = CONCAT('legacy-', `algorithm_id`) WHERE `model_id` IS NULL;
            ALTER TABLE `sys_eval_log` MODIFY COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图像模型标识（统一模型规范 slug）';
            IF (SELECT COUNT(*) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'sys_eval_log' AND INDEX_NAME = 'idx_algorithm_id') > 0 THEN
                ALTER TABLE `sys_eval_log` DROP INDEX `idx_algorithm_id`;
            END IF;
            ALTER TABLE `sys_eval_log` DROP COLUMN `algorithm_id`;
            ALTER TABLE `sys_eval_log` ADD INDEX `idx_model_id` (`model_id`) USING BTREE;
        END IF;

        -- ================= sys_rating =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_rating' AND COLUMN_NAME = 'algorithm_id') > 0 THEN
            ALTER TABLE `sys_rating`
                ADD COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL AFTER `pred_log_id`;
            UPDATE `sys_rating` r JOIN `_algo_slug` s ON s.algo_id = r.algorithm_id SET r.model_id = s.slug;
            UPDATE `sys_rating` SET `model_id` = CONCAT('legacy-', `algorithm_id`) WHERE `model_id` IS NULL;
            ALTER TABLE `sys_rating` MODIFY COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图像模型标识（统一模型规范 slug）';
            IF (SELECT COUNT(*) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'sys_rating' AND INDEX_NAME = 'idx_algorithm_id') > 0 THEN
                ALTER TABLE `sys_rating` DROP INDEX `idx_algorithm_id`;
            END IF;
            ALTER TABLE `sys_rating` DROP COLUMN `algorithm_id`;
            ALTER TABLE `sys_rating` ADD INDEX `idx_model_id` (`model_id`) USING BTREE;
        END IF;

        -- ================= sys_input_history =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_input_history' AND COLUMN_NAME = 'algorithm_id') > 0 THEN
            ALTER TABLE `sys_input_history`
                ADD COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '图像模型标识（统一模型规范 slug）',
                ADD COLUMN `model_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '模型名称（冗余）',
                ADD COLUMN `model_params` json NULL COMMENT '模型参数（JSON）';
            UPDATE `sys_input_history` h JOIN `_algo_slug` s ON s.algo_id = h.algorithm_id SET h.model_id = s.slug;
            UPDATE `sys_input_history` SET `model_id` = CONCAT('legacy-', `algorithm_id`) WHERE `model_id` IS NULL AND `algorithm_id` IS NOT NULL;
            UPDATE `sys_input_history` SET `model_name` = `algorithm_name`, `model_params` = `algorithm_params`;
            ALTER TABLE `sys_input_history`
                DROP COLUMN `algorithm_id`,
                DROP COLUMN `algorithm_name`,
                DROP COLUMN `algorithm_params`;
        END IF;

        -- ================= sys_preset =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_preset' AND COLUMN_NAME = 'algorithm_id') > 0 THEN
            ALTER TABLE `sys_preset`
                ADD COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL AFTER `type`;
            UPDATE `sys_preset` p JOIN `_algo_slug` s ON s.algo_id = p.algorithm_id SET p.model_id = s.slug;
            UPDATE `sys_preset` SET `model_id` = CONCAT('legacy-', `algorithm_id`) WHERE `model_id` IS NULL;
            ALTER TABLE `sys_preset` MODIFY COLUMN `model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '关联图像模型标识（统一模型规范 slug）';
            IF (SELECT COUNT(*) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = 'sys_preset' AND INDEX_NAME = 'idx_algorithm_id') > 0 THEN
                ALTER TABLE `sys_preset` DROP INDEX `idx_algorithm_id`;
            END IF;
            ALTER TABLE `sys_preset` DROP COLUMN `algorithm_id`;
            ALTER TABLE `sys_preset` ADD INDEX `idx_model_id` (`model_id`) USING BTREE;
        END IF;

        -- ================= sys_recommendation =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_recommendation' AND COLUMN_NAME = 'adopted_algorithm_id') > 0 THEN
            ALTER TABLE `sys_recommendation`
                CHANGE COLUMN `top_algorithms` `top_models` json NOT NULL COMMENT '推荐图像模型列表（JSON数组，Top 3：模型 slug 及匹配度）',
                ADD COLUMN `adopted_model_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户采纳的图像模型标识（null:未采纳）' AFTER `feedback`;
            UPDATE `sys_recommendation` r JOIN `_algo_slug` s ON s.algo_id = r.adopted_algorithm_id SET r.adopted_model_id = s.slug;
            UPDATE `sys_recommendation` SET `adopted_model_id` = CONCAT('legacy-', `adopted_algorithm_id`)
                WHERE `adopted_model_id` IS NULL AND `adopted_algorithm_id` IS NOT NULL;
            ALTER TABLE `sys_recommendation` DROP COLUMN `adopted_algorithm_id`;
            UPDATE `sys_recommendation` SET `target_type` = 'model' WHERE `target_type` = 'algorithm';

            -- top_models 内容重建：algorithmId/algorithmName → modelId/modelName，值转 slug
            DROP TEMPORARY TABLE IF EXISTS `_rec_models`;
            CREATE TEMPORARY TABLE `_rec_models` (`rec_id` bigint NOT NULL, `payload` json NOT NULL, PRIMARY KEY (`rec_id`));
            INSERT INTO `_rec_models` (`rec_id`, `payload`)
            SELECT r.id,
                   COALESCE(JSON_ARRAYAGG(JSON_OBJECT(
                       'modelId', COALESCE(s.slug, CONCAT('legacy-', j.aid)),
                       'modelName', j.name,
                       'matchScore', j.score)), JSON_ARRAY())
            FROM `sys_recommendation` r
            JOIN JSON_TABLE(r.top_models, '$[*]' COLUMNS (
                     aid int PATH '$.algorithmId',
                     name varchar(100) PATH '$.algorithmName',
                     score int PATH '$.matchScore')) j ON 1 = 1
            LEFT JOIN `_algo_slug` s ON s.algo_id = j.aid
            GROUP BY r.id;
            UPDATE `sys_recommendation` r JOIN `_rec_models` m ON m.rec_id = r.id SET r.top_models = m.payload;
            DROP TEMPORARY TABLE IF EXISTS `_rec_models`;
        END IF;

        -- ================= sys_recommendation_rule =================
        IF (SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'sys_recommendation_rule' AND COLUMN_NAME = 'algorithm_ids') > 0 THEN
            ALTER TABLE `sys_recommendation_rule`
                CHANGE COLUMN `algorithm_ids` `model_ids` json NOT NULL COMMENT '候选图像模型标识列表（JSON数组，统一模型规范 slug）';

            DROP TEMPORARY TABLE IF EXISTS `_rule_models`;
            CREATE TEMPORARY TABLE `_rule_models` (`rule_id` bigint NOT NULL, `payload` json NOT NULL, PRIMARY KEY (`rule_id`));
            INSERT INTO `_rule_models` (`rule_id`, `payload`)
            SELECT r.id, COALESCE(JSON_ARRAYAGG(COALESCE(s.slug, CONCAT('legacy-', j.aid))), JSON_ARRAY())
            FROM `sys_recommendation_rule` r
            JOIN JSON_TABLE(r.model_ids, '$[*]' COLUMNS (aid int PATH '$')) j ON 1 = 1
            LEFT JOIN `_algo_slug` s ON s.algo_id = j.aid
            GROUP BY r.id;
            UPDATE `sys_recommendation_rule` r JOIN `_rule_models` m ON m.rule_id = r.id SET r.model_ids = m.payload;
            DROP TEMPORARY TABLE IF EXISTS `_rule_models`;
        END IF;

        -- ================= sys_favorite =================
        -- target_type='algorithm' → 'model'，target_id 由算法 id 映射为 sys_ai_model 主键
        UPDATE `sys_favorite` f
            JOIN `_algo_slug` s ON s.algo_id = f.target_id
            JOIN `sys_ai_model` m ON m.model_id = s.slug AND m.deleted = 0
        SET f.target_id = m.id,
            f.target_type = 'model'
        WHERE f.target_type = 'algorithm';

        -- 未映射（算法无权重路径/测试算法）的收藏指向已不存在对象 → 按软删约定（deleted=行id）清理
        UPDATE `sys_favorite` SET `deleted` = `id` WHERE `target_type` = 'algorithm' AND `deleted` = 0;

        -- ================= 会员配额列停用 + 存量折算积分（设计 §9.1） =================
        UPDATE `sys_member_benefit`
        SET `monthly_dehaze_quota` = 0,
            `monthly_derain_quota` = 0,
            `monthly_desnow_quota` = 0,
            `monthly_lowlight_quota` = 0,
            `monthly_super_resolution_quota` = 0,
            `monthly_denoise_quota` = 0,
            `monthly_inpaint_quota` = 0,
            `monthly_evaluate_quota` = 0
        WHERE `deleted` = 0;

        -- 幂等守卫：折算发放仅执行一次（已有折算流水则跳过，避免重跑重复发积分）
        IF (SELECT COUNT(*) FROM `sys_ai_credit_log`
                WHERE `reason` = '阶段3 图像配额折算（当月剩余次数×1积分）' AND `deleted` = 0) = 0 THEN
            DROP TEMPORARY TABLE IF EXISTS `_quota_convert`;
            CREATE TEMPORARY TABLE `_quota_convert` (`user_id` bigint NOT NULL, `credits` bigint NOT NULL, PRIMARY KEY (`user_id`));
            INSERT INTO `_quota_convert` (`user_id`, `credits`)
            SELECT m.user_id,
                   (m.monthly_dehaze_quota - m.monthly_dehaze_used)
                 + (m.monthly_derain_quota - m.monthly_derain_used)
                 + (m.monthly_desnow_quota - m.monthly_desnow_used)
                 + (m.monthly_lowlight_quota - m.monthly_lowlight_used)
                 + (m.monthly_super_resolution_quota - m.monthly_super_resolution_used)
                 + (m.monthly_denoise_quota - m.monthly_denoise_used)
                 + (m.monthly_inpaint_quota - m.monthly_inpaint_used)
            FROM `sys_member` m
            WHERE m.deleted = 0 AND m.status = 1;
            DELETE FROM `_quota_convert` WHERE `credits` <= 0;

            UPDATE `sys_user` u JOIN `_quota_convert` c ON c.user_id = u.id AND u.deleted = 0
            SET u.credits_balance = u.credits_balance + c.credits,
                u.credits_version = u.credits_version + 1;

            INSERT INTO `sys_ai_credit_log`
                (`user_id`, `source`, `amount`, `balance_after`, `reason`, `deleted`, `create_by`, `update_by`)
            SELECT c.user_id, 'admin_adjust', c.credits, u.credits_balance,
                   '阶段3 图像配额折算（当月剩余次数×1积分）', 0, 2, 2
            FROM `_quota_convert` c JOIN `sys_user` u ON u.id = c.user_id;
            DROP TEMPORARY TABLE IF EXISTS `_quota_convert`;
        END IF;

        -- ================= 不可回退：DROP 算法两表 =================
        DROP TABLE IF EXISTS `sys_algorithm`;
        DROP TABLE IF EXISTS `sys_algorithm_version`;

        SELECT '阶段 3 图像链路迁移完成，sys_algorithm(_version) 已 DROP（不可回退点已执行）' AS result;
    END IF;
END$$
DELIMITER ;

CALL `_stage3_image_migrate`();
DROP PROCEDURE IF EXISTS `_stage3_image_migrate`;

-- ---------- 菜单/权限：算法管理目录树与算法选择退役（存量库 DML，与静态种子一致，幂等） ----------
DELETE FROM `sys_role_menu` WHERE `menu_id` IN (29, 30, 31, 32, 33, 34, 35, 36, 37, 55, 65, 113, 114, 160);
DELETE FROM `sys_menu` WHERE `id` IN (29, 30, 31, 32, 33, 34, 35, 36, 37, 55, 65, 113, 114, 160);
-- 推荐规则权限改挂「AI模型管理 → 模型管理」（id 132），保留能力（推荐管理模块按设计保留）
UPDATE `sys_menu` SET `parent_id` = 132, `tree_path` = '0,130,132' WHERE `id` IN (104, 105, 106);

-- ---------- dict：voice_billing 单价键废除（与阶段 2 一致，幂等） ----------
DELETE FROM `sys_dict`
WHERE `type_code` = 'voice_billing'
  AND `name` IN ('asr_credits_per_second', 'tts_credits_per_char')
  AND `deleted` = 0;

-- ---------- xxl-job：会员月度配额重置任务退役（三端实现已删；xxl_job_info 仅存在于 xxl-job-admin 库，
--            业务库缺失时跳过，故用动态 SQL 守卫 + 幂等） ----------
SET @has_xxl_job_info := (SELECT COUNT(*) FROM information_schema.TABLES
                          WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'xxl_job_info');
SET @del_reset_quota_sql := IF(@has_xxl_job_info > 0,
                               'DELETE FROM `xxl_job_info` WHERE `executor_handler` = ''resetMonthlyQuota''',
                               'DO 0');
PREPARE stmt_reset_quota FROM @del_reset_quota_sql;
EXECUTE stmt_reset_quota;
DEALLOCATE PREPARE stmt_reset_quota;

-- 执行后需清 Redis 余额缓存（折算发放后 MySQL 与 Redis 最终一致；TTL 1 小时）：
--   redis-cli -n 1 --scan --pattern 'ai:balance:*' | xargs -r redis-cli -n 1 del
