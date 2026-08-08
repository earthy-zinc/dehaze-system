-- ============================================================
-- 表名: sys_member_benefit
-- 模块: 商业化模块-会员管理
-- ============================================================
-- 设计思路:
-- 会员等级权益配置表，定义四个等级（普通/VIP1/VIP2/SVIP）的默认权益。
-- level_code 唯一索引保证每个等级一条配置。
-- growth_min/growth_max 定义成长值区间，用于自动升降级判断。
-- 权益项覆盖次数配额（去雾/评估/历史保留/批量上限）、AI 对话积分配额（日/月）、多模态视觉读取频次、
-- 处理优先级、功能解锁开关。
-- 功能解锁项使用 tinyint(0/1) 而非布尔，与系统其他表风格一致。
-- 套餐购买时从此表读取等级权益，套餐可自定义覆盖（见 sys_package.benefit_overrides）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_member_benefit`;
CREATE TABLE `sys_member_benefit`
(
    `id`                     bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `level_code`             varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '会员等级(level_0/level_1/level_2/level_3)',
    `level_name`             varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '等级名称',
    `growth_min`             bigint                                                         NOT NULL DEFAULT 0 COMMENT '成长值下限',
    `growth_max`             bigint                                                         NOT NULL DEFAULT 0 COMMENT '成长值上限（0表示无上限）',
    `monthly_dehaze_quota`   int                                                            NOT NULL DEFAULT 0 COMMENT '月度去雾次数配额',
    `monthly_evaluate_quota` int                                                            NOT NULL DEFAULT 0 COMMENT '月度评估次数配额',
    `history_retention`     int                                                            NOT NULL DEFAULT 0 COMMENT '历史记录保留条数',
    `batch_limit`            int                                                            NOT NULL DEFAULT 0 COMMENT '批量处理上限（张）',
    `priority`               tinyint                                                        NOT NULL DEFAULT 1 COMMENT '处理优先级(1:普通;2:优先;3:高优先;4:最高)',
    `advanced_params`        tinyint                                                        NOT NULL DEFAULT 0 COMMENT '高级参数调节(0:关闭;1:开启)',
    `hd_export`              tinyint                                                        NOT NULL DEFAULT 0 COMMENT '高清图导出(0:关闭;1:开启)',
    `report_export`          tinyint                                                        NOT NULL DEFAULT 0 COMMENT '对比报告导出(0:关闭;1:开启)',
    `batch_download`         tinyint                                                        NOT NULL DEFAULT 0 COMMENT '批量打包下载(0:关闭;1:开启)',
    `ai_credits_daily`       bigint                                                         NOT NULL DEFAULT 0 COMMENT 'AI对话日限额(积分/天，每日0点重置)',
    `ai_credits_monthly`     bigint                                                         NOT NULL DEFAULT 0 COMMENT 'AI对话月限额(积分/月，每月1日重置)',
    `multimodal_limit`       int                                                            NOT NULL DEFAULT 0 COMMENT '单会话多模态视觉读取频次上限',
    `sort`                   int                                                            NOT NULL DEFAULT 0 COMMENT '排序值',
    `status`                 tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`                tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`            datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`              bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`              bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_level_code` (`level_code`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '会员等级权益配置表'
  ROW_FORMAT = DYNAMIC;
