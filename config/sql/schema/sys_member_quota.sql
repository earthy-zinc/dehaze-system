-- ============================================================
-- 表名: sys_member_quota
-- 模块: 商业化模块-会员管理
-- ============================================================
-- 设计思路:
-- 会员月度配额历史表，每月一条记录，用于追溯历史月份配额使用情况。
-- 与 sys_member 当月配额字段配合：sys_member 存当月实时数据，此表按月归档。
-- 归档 8 类任务配额（dehaze/derain/desnow/lowlight/super_resolution/denoise/inpaint/evaluate），
--   与 sys_member 当月配额字段、sys_member_benefit 权益配置列对齐。
-- (user_id, quota_month) 唯一索引保证每月一条。
-- 定时任务每月1日执行：先将上月数据写入此表，再重置 sys_member 的当月配额字段。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_member_quota`;
CREATE TABLE `sys_member_quota`
(
    `id`                     bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`                bigint                                                         NOT NULL COMMENT '用户ID',
    `quota_month`            int                                                            NOT NULL COMMENT '配额月份（格式yyyyMM）',
    `level_code`             varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '当月会员等级',
    `dehaze_quota`           int                                                            NOT NULL DEFAULT 0 COMMENT '当月去雾配额',
    `dehaze_used`            int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用去雾次数',
    `derain_quota`           int                                                            NOT NULL DEFAULT 0 COMMENT '当月去雨配额',
    `derain_used`            int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用去雨次数',
    `desnow_quota`           int                                                            NOT NULL DEFAULT 0 COMMENT '当月去雪配额',
    `desnow_used`            int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用去雪次数',
    `lowlight_quota`         int                                                            NOT NULL DEFAULT 0 COMMENT '当月低光增强配额',
    `lowlight_used`          int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用低光增强次数',
    `super_resolution_quota` int                                                            NOT NULL DEFAULT 0 COMMENT '当月超分辨率配额',
    `super_resolution_used`  int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用超分辨率次数',
    `denoise_quota`          int                                                            NOT NULL DEFAULT 0 COMMENT '当月去噪配额',
    `denoise_used`           int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用去噪次数',
    `inpaint_quota`          int                                                            NOT NULL DEFAULT 0 COMMENT '当月图像修复配额',
    `inpaint_used`           int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用图像修复次数',
    `evaluate_quota`         int                                                            NOT NULL DEFAULT 0 COMMENT '当月评估配额',
    `evaluate_used`          int                                                            NOT NULL DEFAULT 0 COMMENT '当月已用评估次数',
    `reset_time`             datetime                                                       NOT NULL COMMENT '配额重置时间',
    `deleted`                tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`              bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`              bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_month` (`user_id`, `quota_month`) USING BTREE,
    INDEX `idx_quota_month` (`quota_month`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '会员月度配额历史表'
  ROW_FORMAT = DYNAMIC;
