-- ============================================================
-- 表名: sys_ai_refund
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- AI 积分误扣退款申请表，承载退款申请的状态流转（1待审核→2已通过/3已驳回），
-- 支撑需求规格 §2.3 退款补偿流程（用户/系统申请 → 管理员审核 → 余额回补 → 流水记录）。
-- billing_id 关联原计费记录（sys_ai_billing.id），用于审计追溯与"不重复申请"校验
-- （同一 billing_id 只允许存在一条待审核（status=1）申请）。
-- amount 为退款积分数；status 状态机：1(待审核)/2(已通过)/3(已驳回)，编码统一遵循 §5.4 tinyint 规范。
-- create_by 记录申请人，auditor_id 记录审核人，audit_remark 记录审核意见；审核通过后由 RefundService
--   回补余额并写 sys_ai_credit_log(source=refund)。
-- 只追加表，不逻辑删除（退款申请为审计记录），历史记录通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_refund`;
CREATE TABLE `sys_ai_refund`
(
    `id`           bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`      bigint                                                          NOT NULL COMMENT '用户ID(关联sys_user.id)',
    `billing_id`   bigint                                                          NOT NULL COMMENT '原计费记录ID(关联sys_ai_billing.id)',
    `amount`       int                                                             NOT NULL COMMENT '退款积分数',
    `reason`       varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '退款原因',
    `status`       tinyint                                                         NOT NULL DEFAULT 1 COMMENT '退款状态(1:待审核;2:已通过;3:已驳回)',
    `auditor_id`   bigint                                                          NULL DEFAULT NULL COMMENT '审核人ID',
    `audit_remark` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '审核意见',
    `create_by`    bigint                                                          NULL DEFAULT NULL COMMENT '申请人ID(用户申请退款时记录)',
    `create_time`  datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_status` (`user_id`, `status`) USING BTREE,
    INDEX `idx_billing_id` (`billing_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI计费退款申请表'
  ROW_FORMAT = DYNAMIC;
