-- 2026-09-12 模型可用性测试：sys_ai_model 加最近测试结果三列
-- 开发库增量执行（测试库由 conftest 全量重建自动生效）
ALTER TABLE `sys_ai_model`
    ADD COLUMN `last_test_status` tinyint NOT NULL DEFAULT 0 COMMENT '最近可用性测试状态(0:未测试;1:可用;2:不可用)' AFTER `status`,
    ADD COLUMN `last_test_at` datetime NULL DEFAULT NULL COMMENT '最近可用性测试时间' AFTER `last_test_status`,
    ADD COLUMN `last_test_error` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '最近可用性测试错误信息(含HTTP状态与延迟)' AFTER `last_test_at`;
