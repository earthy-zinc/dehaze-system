SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================
-- sys_recommendation_rule 种子数据（系统预置推荐规则）
-- 说明：
--   - rule_name 统一以 sys_ 前缀开头，与测试自建规则（测试规则_xxx）区分。
--   - 推荐匹配按 scene_type 精确命中，model_ids 指向已启用图像模型（此处为平台默认模型 aecrnet）。
--   - 覆盖全部有效场景类型，保证任意分析结果都能命中推荐，激活闭环。
--   - 幂等：本文件约定由装载脚本先清理 sys_% 规则再插入；重复装载不产生重复数据。
-- ------------------------------------------------------------
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_urban', 'urban', '["aecrnet"]', 80, 1, 0, 2, 2);
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_landscape', 'landscape', '["aecrnet"]', 70, 1, 0, 2, 2);
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_building', 'building', '["aecrnet"]', 60, 1, 0, 2, 2);
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_night', 'night', '["aecrnet"]', 50, 1, 0, 2, 2);
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_backlight', 'backlight', '["aecrnet"]', 40, 1, 0, 2, 2);
insert into sys_recommendation_rule (rule_name, scene_type, model_ids, weight, enabled, deleted, create_by, update_by)
values ('sys_rule_indoor', 'indoor', '["aecrnet"]', 30, 1, 0, 2, 2);

SET FOREIGN_KEY_CHECKS = 1;
