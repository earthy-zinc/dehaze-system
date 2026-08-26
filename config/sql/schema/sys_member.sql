-- ============================================================
-- 表名: sys_member
-- 模块: 商业化模块-会员管理
-- ============================================================
-- 设计思路:
-- 会员表，与 sys_user 一对一，用户注册时初始化。
-- level_code 记录当前会员等级（level_0/level_1/level_2/level_3），level_source 区分等级来源（成长值达标/购买套餐/管理员调整）。
-- growth_value 记录成长值，expire_time 为套餐到期时间（NULL 表示由成长值维持等级）。
-- 累计消费 total_consumption 用于运营分析，按支付金额累计（单位：分）。
-- status 字段控制会员冻结状态（1:正常;0:冻结），冻结时权益暂停但不影响等级。
-- 当月配额字段（monthly_{taskType}_quota/used，覆盖 8 类任务：dehaze/derain/desnow/lowlight/
--   super_resolution/denoise/inpaint/evaluate）由定时任务每月1日重置，避免单独建配额表；
--   与 sys_member_benefit 权益配置 8 类配额列对齐（任务类型为封闭枚举，宽表读性能最优）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_member`;
CREATE TABLE `sys_member`
(
    `id`                     bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`                bigint                                                         NOT NULL COMMENT '用户ID（关联sys_user.id）',
    `level_code`             varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'level_0' COMMENT '会员等级(level_0:普通用户;level_1:VIP1;level_2:VIP2;level_3:SVIP)',
    `level_source`            varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'growth' COMMENT '等级来源(growth:成长值达标;purchase:套餐购买;admin:管理员调整)',
    `growth_value`           bigint                                                         NOT NULL DEFAULT 0 COMMENT '成长值',
    `total_consumption`      bigint                                                         NOT NULL DEFAULT 0 COMMENT '累计消费金额（单位：分）',
    `expire_time`            datetime                                                       NULL DEFAULT NULL COMMENT '套餐到期时间（NULL表示成长值维持）',
    `become_member_time`     datetime                                                       NULL DEFAULT NULL COMMENT '首次成为会员时间',
    `monthly_dehaze_quota`   int                                                            NOT NULL DEFAULT 0 COMMENT '本月去雾配额',
    `monthly_dehaze_used`    int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用去雾次数',
    `monthly_derain_quota`   int                                                            NOT NULL DEFAULT 0 COMMENT '本月去雨配额',
    `monthly_derain_used`    int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用去雨次数',
    `monthly_desnow_quota`   int                                                            NOT NULL DEFAULT 0 COMMENT '本月去雪配额',
    `monthly_desnow_used`    int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用去雪次数',
    `monthly_lowlight_quota` int                                                            NOT NULL DEFAULT 0 COMMENT '本月低光增强配额',
    `monthly_lowlight_used`  int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用低光增强次数',
    `monthly_super_resolution_quota` int                                                   NOT NULL DEFAULT 0 COMMENT '本月超分辨率配额',
    `monthly_super_resolution_used`  int                                                   NOT NULL DEFAULT 0 COMMENT '本月已用超分辨率次数',
    `monthly_denoise_quota`  int                                                            NOT NULL DEFAULT 0 COMMENT '本月去噪配额',
    `monthly_denoise_used`   int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用去噪次数',
    `monthly_inpaint_quota`  int                                                            NOT NULL DEFAULT 0 COMMENT '本月图像修复配额',
    `monthly_inpaint_used`   int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用图像修复次数',
    `monthly_evaluate_quota` int                                                            NOT NULL DEFAULT 0 COMMENT '本月评估配额',
    `monthly_evaluate_used`  int                                                            NOT NULL DEFAULT 0 COMMENT '本月已用评估次数',
    `quota_reset_month`      int                                                            NULL DEFAULT NULL COMMENT '配额所属月份（格式yyyyMM）',
    `status`                 tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:正常;0:冻结)',
    `frozen_reason`          varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '冻结原因',
    `frozen_time`            datetime                                                       NULL DEFAULT NULL COMMENT '冻结时间',
    `deleted`                tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`              bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`              bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_id` (`user_id`) USING BTREE,
    INDEX `idx_level_code` (`level_code`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE,
    INDEX `idx_expire_time` (`expire_time`) USING BTREE,
    INDEX `idx_quota_reset_month` (`quota_reset_month`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '会员信息表'
  ROW_FORMAT = DYNAMIC;
