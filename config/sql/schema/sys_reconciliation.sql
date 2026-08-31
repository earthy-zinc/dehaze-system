-- ============================================================
-- 表名: sys_reconciliation
-- 模块: 商业化模块-订单管理
-- ============================================================
-- 设计思路:
-- 渠道对账差异表（需求规格 §5.1 / F-OM-016）。每日对账任务取前一日支付成功流水，
-- 与微信/支付宝渠道账单按支付流水号逐单核对，差异（系统多单 system_only /
-- 渠道多单 channel_only / 金额不符 amount_mismatch）落本表，由运营跟进处理。
-- 同一 (recon_date, flow_no) 唯一；重跑对账时按对账日全量重写。
-- system_amount/channel_amount 依差异类型可空（单侧缺失时对侧为空）。
-- 标准逻辑删除（类别③，流水追溯型，删除后流水号不可复用）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_reconciliation`;
CREATE TABLE `sys_reconciliation`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `recon_date`      date                                                           NOT NULL COMMENT '对账日期',
    `channel`         varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付渠道(wechat:微信;alipay:支付宝)',
    `flow_no`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '支付流水号',
    `order_no`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '关联订单号(渠道多单时来自渠道账单)',
    `system_amount`   bigint                                                         NULL DEFAULT NULL COMMENT '系统侧金额（单位：分，渠道多单时为空）',
    `channel_amount`  bigint                                                         NULL DEFAULT NULL COMMENT '渠道侧金额（单位：分，系统多单时为空）',
    `diff_type`       varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '差异类型(amount_mismatch:金额不符;system_only:系统多单;channel_only:渠道多单)',
    `status`          tinyint                                                        NOT NULL DEFAULT 0 COMMENT '处理状态(0:未处理;1:已处理)',
    `handle_remark`   varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '处理备注',
    `handle_time`     datetime                                                       NULL DEFAULT NULL COMMENT '处理时间',
    `handler_id`      bigint                                                         NULL DEFAULT NULL COMMENT '处理人ID',
    `deleted`         tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_recon_date_flow_no` (`recon_date`, `flow_no`) USING BTREE,
    INDEX `idx_status` (`status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '渠道对账差异表'
  ROW_FORMAT = DYNAMIC;
