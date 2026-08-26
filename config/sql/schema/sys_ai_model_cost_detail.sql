-- ============================================================
-- 表名: sys_ai_model_cost_detail
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- 成本单价档位明细表（与版本主表 sys_ai_model_cost 构成主从结构），与用户售价档位明细表结构完全对称。
-- 每行 = 一个 token 类型 × 一个上下文分段 × 一个时段档位的采购单价（元/百万 token）。
-- token_type：input(输入未命中)/cached(缓存命中)/output(输出)。
-- time_slot：peak(高峰)/idle(空闲)，时段判定规则全局配置（ai.billing.peak-hours），未来可扩展时段枚举。
-- min_tokens/max_tokens：上下文分段区间（按本次调用输入总 token 数定位），无分段模型一行即可。
-- 档位随价格版本（sys_ai_model_cost）整体管理；同 price_id 内 (token_type, time_slot, 分段区间) 不得重叠。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_model_cost_detail`;
CREATE TABLE `sys_ai_model_cost_detail`
(
    `id`             bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `price_id`       bigint                                                          NOT NULL COMMENT '价格版本ID(关联sys_ai_model_cost.id)',
    `token_type`     varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '计费类型(input:输入未命中;cached:缓存命中;output:输出)',
    `time_slot`      varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '时段档位(peak:高峰;idle:空闲)',
    `min_tokens`     bigint                                                          NOT NULL DEFAULT 0 COMMENT '上下文分段下界(按本次输入总token数,含缓存命中)',
    `max_tokens`     bigint                                                          NULL DEFAULT NULL COMMENT '上下文分段上界(NULL表示不限,区间为[min_tokens,max_tokens))',
    `unit_price`     decimal(12, 4)                                                  NOT NULL DEFAULT 0.0000 COMMENT '单价(元/百万token)',
    `deleted`        tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`    datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`      bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_price_token_time_range` (`price_id`, `token_type`, `time_slot`, `min_tokens`, `max_tokens`) USING BTREE,
    INDEX `idx_price` (`price_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI模型成本单价档位明细表'
  ROW_FORMAT = DYNAMIC;
