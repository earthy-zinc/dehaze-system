-- ============================================================
-- 唯一键治理：唯一索引感知逻辑删除（方案A定稿）
-- 语义：deleted = 0 未删除；deleted > 0 已删除（值为删除时的行 id）
-- 执行顺序：①数据修复（历史软删行 deleted=1 → deleted=id，避免新唯一键同键冲突）
--          ②全库 deleted 类型升级 tinyint→bigint
--          ③37 表唯一键追加 deleted 列（索引名保持不变）
-- 注意：一次性执行，不可重复执行（DROP INDEX 无 IF EXISTS，重跑会报错）
-- ============================================================

-- ① 数据修复：历史软删行补写行 id
UPDATE `sys_ai_agent` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_agent_endpoint` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_agent_eval_dataset` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_conversation` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_credit_log` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_mcp_server` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_memory` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_message` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_message_feedback` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_model` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_model_cost` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_model_cost_detail` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_model_price` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_model_price_detail` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_provider` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_schedule` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_ai_skill` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_algorithm` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_algorithm_version` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_announcement` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_auto_renew` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_balance` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_balance_log` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_balance_refund` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_coupon` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_dataset` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_dataset_item` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_dept` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_dict` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_dict_type` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_favorite` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_feedback` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_feedback_reply` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_file` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_item_file` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_knowledge_base` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_knowledge_document` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_knowledge_test_set` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_member` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_member_benefit` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_member_growth_log` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_member_quota` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_menu` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_message` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_message_template` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_notification_setting` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_order` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_package` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_payment_record` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_promotion` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_rating` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_recharge` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_recommendation_rule` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_reconciliation` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_refund_record` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_role` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_user` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_user_coupon` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_voice_hotword` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_voice_model` SET `deleted` = `id` WHERE `deleted` = 1;
UPDATE `sys_voice_provider` SET `deleted` = `id` WHERE `deleted` = 1;

-- ② 全库 deleted 类型升级（与 schema 一致，语义统一为行 id）
ALTER TABLE `sys_ai_agent` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_agent_endpoint` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_agent_eval_dataset` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_conversation` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_credit_log` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_mcp_server` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_memory` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_message` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_message_feedback` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_model` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_model_cost` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_model_cost_detail` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_model_price` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_model_price_detail` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_provider` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_schedule` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_ai_skill` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_algorithm` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_algorithm_version` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_announcement` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_auto_renew` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_balance` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_balance_log` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_balance_refund` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_coupon` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_dataset` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_dataset_item` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_dept` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_dict` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_dict_type` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_favorite` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_feedback` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_feedback_reply` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_file` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_item_file` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_knowledge_base` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_knowledge_document` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_knowledge_test_set` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_member` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_member_benefit` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_member_growth_log` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_member_quota` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_menu` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_message` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_message_template` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_notification_setting` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_order` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_package` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_payment_record` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_promotion` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_rating` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_recharge` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_recommendation_rule` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_reconciliation` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_refund_record` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_role` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_user` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_user_coupon` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_voice_hotword` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_voice_model` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';
ALTER TABLE `sys_voice_provider` MODIFY `deleted` bigint NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)';

-- ③ 唯一键追加 deleted 列（37 表，索引名不变）
ALTER TABLE `sys_ai_agent` DROP INDEX `uk_agent_code`, ADD UNIQUE INDEX `uk_agent_code` (`agent_code`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_agent_endpoint` DROP INDEX `uk_base_url`, ADD UNIQUE INDEX `uk_base_url` (`base_url`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_agent_eval_dataset` DROP INDEX `uk_agent_dataset_type`, ADD UNIQUE INDEX `uk_agent_dataset_type` (`agent_id`, `dataset_type`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_mcp_server` DROP INDEX `uk_name`, ADD UNIQUE INDEX `uk_name` (`name`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_message_feedback` DROP INDEX `uk_message_user`, ADD UNIQUE INDEX `uk_message_user` (`message_id`, `user_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_model` DROP INDEX `uk_model_provider`, ADD UNIQUE INDEX `uk_model_provider` (`model_id`, `provider_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_model_cost` DROP INDEX `uk_model_provider_version`, ADD UNIQUE INDEX `uk_model_provider_version` (`model_id`, `provider_id`, `price_version`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_model_cost_detail` DROP INDEX `uk_price_token_time_range`, ADD UNIQUE INDEX `uk_price_token_time_range` (`price_id`, `token_type`, `time_slot`, `min_tokens`, `max_tokens`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_model_price` DROP INDEX `uk_model_provider_version`, ADD UNIQUE INDEX `uk_model_provider_version` (`model_id`, `provider_id`, `price_version`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_model_price_detail` DROP INDEX `uk_price_token_time_range`, ADD UNIQUE INDEX `uk_price_token_time_range` (`price_id`, `token_type`, `time_slot`, `min_tokens`, `max_tokens`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_provider` DROP INDEX `uk_provider_code`, ADD UNIQUE INDEX `uk_provider_code` (`provider_code`, `deleted`) USING BTREE;

ALTER TABLE `sys_ai_skill` DROP INDEX `uk_name`, ADD UNIQUE INDEX `uk_name` (`name`, `deleted`) USING BTREE;

ALTER TABLE `sys_algorithm_version` DROP INDEX `uk_algo_version`, ADD UNIQUE INDEX `uk_algo_version` (`algorithm_id`, `version`, `deleted`) USING BTREE;

ALTER TABLE `sys_auto_renew` DROP INDEX `uk_user_package`, ADD UNIQUE INDEX `uk_user_package` (`user_id`, `package_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_balance` DROP INDEX `uk_user_id`, ADD UNIQUE INDEX `uk_user_id` (`user_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_balance_refund` DROP INDEX `uk_refund_no`, ADD UNIQUE INDEX `uk_refund_no` (`refund_no`, `deleted`) USING BTREE;

ALTER TABLE `sys_dict` DROP INDEX `uk_type_name`, ADD UNIQUE INDEX `uk_type_name` (`type_code`, `name`, `deleted`) USING BTREE;

ALTER TABLE `sys_dict_type` DROP INDEX `uk_code`, ADD UNIQUE INDEX `uk_code` (`code`, `deleted`) USING BTREE;

ALTER TABLE `sys_favorite` DROP INDEX `uk_user_target`, ADD UNIQUE INDEX `uk_user_target` (`user_id`, `target_type`, `target_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_file` DROP INDEX `uk_md5`, ADD UNIQUE INDEX `uk_md5` (`md5`, `deleted`) USING BTREE;

ALTER TABLE `sys_member` DROP INDEX `uk_user_id`, ADD UNIQUE INDEX `uk_user_id` (`user_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_member_benefit` DROP INDEX `uk_level_code`, ADD UNIQUE INDEX `uk_level_code` (`level_code`, `deleted`) USING BTREE;

ALTER TABLE `sys_member_quota` DROP INDEX `uk_user_month`, ADD UNIQUE INDEX `uk_user_month` (`user_id`, `quota_month`, `deleted`) USING BTREE;

ALTER TABLE `sys_message` DROP INDEX `uk_biz_dedup`, ADD UNIQUE INDEX `uk_biz_dedup` (`biz_module`, `biz_id`, `recipient_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_message_template` DROP INDEX `uk_code`, ADD UNIQUE INDEX `uk_code` (`code`, `deleted`) USING BTREE;

ALTER TABLE `sys_notification_setting` DROP INDEX `uk_user_id`, ADD UNIQUE INDEX `uk_user_id` (`user_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_order` DROP INDEX `uk_order_no`, ADD UNIQUE INDEX `uk_order_no` (`order_no`, `deleted`) USING BTREE;

ALTER TABLE `sys_package` DROP INDEX `uk_name`, ADD UNIQUE INDEX `uk_name` (`name`, `deleted`) USING BTREE;

ALTER TABLE `sys_payment_record` DROP INDEX `uk_payment_no`, ADD UNIQUE INDEX `uk_payment_no` (`payment_no`, `deleted`) USING BTREE;

ALTER TABLE `sys_rating` DROP INDEX `uk_pred_log_id`, ADD UNIQUE INDEX `uk_pred_log_id` (`pred_log_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_recharge` DROP INDEX `uk_recharge_no`, ADD UNIQUE INDEX `uk_recharge_no` (`recharge_no`, `deleted`) USING BTREE;
ALTER TABLE `sys_recharge` DROP INDEX `uk_channel_payment_no`, ADD UNIQUE INDEX `uk_channel_payment_no` (`channel_payment_no`, `deleted`) USING BTREE;

ALTER TABLE `sys_reconciliation` DROP INDEX `uk_recon_date_flow_no`, ADD UNIQUE INDEX `uk_recon_date_flow_no` (`recon_date`, `flow_no`, `deleted`) USING BTREE;

ALTER TABLE `sys_refund_record` DROP INDEX `uk_refund_no`, ADD UNIQUE INDEX `uk_refund_no` (`refund_no`, `deleted`) USING BTREE;
ALTER TABLE `sys_refund_record` DROP INDEX `uk_order_id`, ADD UNIQUE INDEX `uk_order_id` (`order_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_role` DROP INDEX `uk_name`, ADD UNIQUE INDEX `uk_name` (`name`, `deleted`) USING BTREE;
ALTER TABLE `sys_role` DROP INDEX `uk_code`, ADD UNIQUE INDEX `uk_code` (`code`, `deleted`) USING BTREE;

ALTER TABLE `sys_user` DROP INDEX `uk_username`, ADD UNIQUE INDEX `uk_username` (`username`, `deleted`) USING BTREE;

ALTER TABLE `sys_voice_model` DROP INDEX `uk_model_provider`, ADD UNIQUE INDEX `uk_model_provider` (`model_id`, `provider_id`, `deleted`) USING BTREE;

ALTER TABLE `sys_voice_provider` DROP INDEX `uk_provider_engine`, ADD UNIQUE INDEX `uk_provider_engine` (`provider_code`, `engine_type`, `deleted`) USING BTREE;

