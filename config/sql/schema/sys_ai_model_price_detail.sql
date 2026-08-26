-- ============================================================
-- 表名: sys_ai_model_price_detail
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- 用户售价档位明细表（与版本主表 sys_ai_model_price 构成主从结构）。
-- 每行 = 一个 token 类型 × 一个上下文分段 × 一个时段档位的单价。
-- token_type 区分计费对象：input(输入未命中)/cached(缓存命中)/output(输出)。
-- time_slot 区分时段档位：peak(高峰)/idle(空闲)，时段判定规则全局配置（ai.billing.peak-hours）；
--   未来新增时段档位仅扩展枚举，不改表结构。
-- min_tokens/max_tokens 定义上下文分段区间（按本次调用输入总 token 数定位）：
--   无分段模型 min_tokens=0、max_tokens=NULL 一行即可；分段模型拆多行（如 0~200K、200K~∞）。
-- 换算时按 调用时刻时段 + 本次输入总长度 + token 类型 三维匹配单价行；
--   不分时段的模型 peak/idle 两行填相同单价。
-- 档位随价格版本（sys_ai_model_price）整体管理，不独立停用；
--   同 price_id 内 (token_type, time_slot, 分段区间) 不得重叠。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_model_price_detail`;
CREATE TABLE `sys_ai_model_price_detail`
(
    `id`             bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `price_id`       bigint                                                          NOT NULL COMMENT '价格版本ID(关联sys_ai_model_price.id)',
    `token_type`     varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '计费类型(input:输入未命中;cached:缓存命中;output:输出)',
    `time_slot`      varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '时段档位(peak:高峰;idle:空闲)',
    `min_tokens`     bigint                                                          NOT NULL DEFAULT 0 COMMENT '上下文分段下界(按本次输入总token数,含缓存命中)',
    `max_tokens`     bigint                                                          NULL DEFAULT NULL COMMENT '上下文分段上界(NULL表示不限,区间为[min_tokens,max_tokens))',
    `unit_price`     decimal(12, 4)                                                  NOT NULL DEFAULT 0.0000 COMMENT '单价(积分/百万token)',
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
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI模型用户售价档位明细表'
  ROW_FORMAT = DYNAMIC;
