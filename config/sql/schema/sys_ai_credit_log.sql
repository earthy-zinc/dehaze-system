-- ============================================================
-- 表名: sys_ai_credit_log
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- 积分余额变动流水表，记录用户积分余额账户的每一笔变动（积分卡到账/VIP赠送/试用/扣减/回退/调整），
-- 支撑余额审计追溯与账单生成。
-- source 区分变动来源：recharge(积分卡到账,关联积分卡订单)/vip_gift(VIP赠送)/trial(试用)/admin_adjust(管理员调整)/
--   refund(回退/补偿)/consume(消耗扣减)/vip_gift_expire(VIP赠送月末清零)。
-- amount 正数表示增加、负数表示扣减；积分语义为整数（计费换算四舍五入到整数），统一使用 bigint。
-- balance_after 记录变动后的账户余额。
-- related_id 关联业务记录（如计费记录ID/积分卡订单ID）；operator_id 为操作人，NULL 表示系统自动。
-- 只追加表，不逻辑删除（流水不可删），过期数据通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_credit_log`;
CREATE TABLE `sys_ai_credit_log`
(
    `id`            bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`       bigint                                                          NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `source`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '变动来源(recharge;vip_gift;trial;admin_adjust;refund;consume;vip_gift_expire)',
    `amount`        bigint                                                          NOT NULL COMMENT '变动积分数(正数增加;负数扣减)',
    `balance_after` bigint                                                          NOT NULL COMMENT '变动后账户余额',
    `related_id`    bigint                                                          NULL DEFAULT NULL COMMENT '关联业务记录ID(如计费记录ID/积分卡订单ID)',
    `reason`        varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '变动原因',
    `operator_id`   bigint                                                          NULL DEFAULT NULL COMMENT '积分变动业务操作人ID(人工调整/客服补偿场景记录;系统自动为NULL)',
    `deleted`       tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`   datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`     bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_create_time` (`user_id`, `create_time`) USING BTREE,
    INDEX `idx_source` (`source`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '积分余额变动流水表'
  ROW_FORMAT = DYNAMIC;
