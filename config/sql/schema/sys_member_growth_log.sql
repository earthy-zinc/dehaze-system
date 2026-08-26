-- ============================================================
-- 表名: sys_member_growth_log
-- 模块: 商业化模块-会员管理
-- ============================================================
-- 设计思路:
-- 成长值流水表，记录每次成长值变动明细。
-- change_type 标识变动来源（图像处理/评估/评价/签到/AI对话/消费/退款扣减/管理员调整/连续签到奖励）。
-- change_value 允许正负，balance 记录变动后余额便于核对。
-- related_id 关联业务ID（订单号/任务ID/签到记录ID），便于溯源。
-- operator_id 仅在管理员调整时记录，其他行为为用户自身操作。
-- 流水记录使用逻辑删除（deleted 字段），三端实体（Java/Python/Go）均映射该字段。
-- 复合索引 (user_id, create_time) 优化用户成长值明细分页查询。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_member_growth_log`;
CREATE TABLE `sys_member_growth_log`
(
    `id`            bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`       bigint                                                         NOT NULL COMMENT '用户ID',
    `change_type`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '变动类型(process:图像处理;evaluate:评估;rating:评价;sign_in:签到;sign_in_bonus:连续签到奖励;consume:消费;ai_consume:AI对话激励;refund_deduct:退款扣减;admin_adjust:管理员调整)',
    `change_value`  int                                                            NOT NULL COMMENT '变动值（正数增加/负数扣减）',
    `balance`       bigint                                                         NOT NULL COMMENT '变动后成长值余额',
    `related_id`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '关联业务ID（订单号/任务ID/签到记录ID）',
    `reason`        varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '变动原因',
    `operator_id`   bigint                                                         NULL DEFAULT NULL COMMENT '操作人ID（仅管理员调整时记录）',
    `deleted`       tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`   datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`     bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id_create_time` (`user_id`, `create_time`) USING BTREE,
    INDEX `idx_change_type` (`change_type`) USING BTREE,
    INDEX `idx_related_id` (`related_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '成长值流水表'
  ROW_FORMAT = DYNAMIC;
