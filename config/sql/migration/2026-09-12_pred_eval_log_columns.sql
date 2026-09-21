-- ============================================================
-- 预测/评估日志表列扩展（C2+C3）
-- ① sys_pred_log 新增 recommended_by：三端 predict 接收 recommendedBy 落库，
--    推荐效果报表采纳率口径 = 有 recommended_by 的预测记录数 / 推荐总数
-- ② sys_eval_log 新增 task_type：区分效果评估(evaluation)与对比报告(report)任务，
--    GET /evaluation/metrics 仅返回 task_type='evaluation'，不再混入报告行
-- 可重复执行性：ADD COLUMN 无 IF NOT EXISTS（MySQL 8.0），重跑会报错
-- ============================================================

-- ① 预测日志：推荐来源
ALTER TABLE `sys_pred_log`
    ADD COLUMN `recommended_by` bigint NULL DEFAULT NULL COMMENT '推荐来源：推荐记录ID（推荐管理模块，用于追踪推荐采纳率）' AFTER `pred_url`,
    ADD INDEX `idx_recommended_by` (`recommended_by`) USING BTREE;

-- ② 评估日志：任务类型（存量行默认 evaluation，报告行按 result 特征回填）
ALTER TABLE `sys_eval_log`
    ADD COLUMN `task_type` varchar(20) NOT NULL DEFAULT 'evaluation' COMMENT '任务类型(evaluation:效果评估;report:对比报告)' AFTER `status`,
    ADD INDEX `idx_task_type` (`task_type`) USING BTREE;

-- 存量对比报告行回填：completed 报告行 result 为 {"reportHtml":...,"generatedAt":...} 包装对象
UPDATE `sys_eval_log` SET `task_type` = 'report' WHERE `result` LIKE '%"reportHtml"%';
