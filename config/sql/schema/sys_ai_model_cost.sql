-- ============================================================
-- 表名: sys_ai_model_cost
-- 模块: 基础模块-AI计费管理
-- ============================================================
-- 设计思路:
-- 模型成本单价版本主表（供应商采购价），具体单价以"档位明细行"表达（见 sys_ai_model_cost_tier），
--   与用户售价表 sys_ai_model_price 结构完全对称，共用价格版本选择与时段判定逻辑。
-- model_id 关联 sys_ai_model.model_id（业务键），provider_id 关联 sys_ai_provider.id，
--   (model_id, provider_id) 对齐 sys_ai_model 的联合唯一键，唯一定位模型。
-- price_version 为价格版本号，供应商调价时新增版本（同 model_id+provider_id 内递增），
--   历史成本按调用时刻对应的价格版本核算（effective_from <= t < effective_to），对账有据。
-- currency 为计费币种（CNY/USD），核算时统一折算为人民币。
-- 只读引用于 AI 对话模块（供应商/模型调价评估），成本核算由 AI 计费模块完成。
-- 配置类表，使用逻辑删除；(model_id, provider_id, price_version) 联合唯一，删除后不可复用（类别②）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_model_cost`;
CREATE TABLE `sys_ai_model_cost`
(
    `id`             bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `model_id`       varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL COMMENT '模型标识(关联sys_ai_model.model_id)',
    `provider_id`    bigint                                                          NOT NULL COMMENT '供应商ID(关联sys_ai_provider.id)',
    `price_version`  int                                                             NOT NULL DEFAULT 1 COMMENT '价格版本号(供应商调价递增,同模型同供应商内唯一)',
    `currency`       varchar(8) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NOT NULL DEFAULT 'CNY' COMMENT '计费币种(CNY/USD)',
    `effective_from` datetime                                                        NOT NULL COMMENT '价格版本生效时间',
    `effective_to`   datetime                                                        NULL DEFAULT NULL COMMENT '价格版本失效时间(NULL表示当前版本)',
    `status`         tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:生效;0:停用)',
    `deleted`        tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`      bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`      bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`    datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_model_provider_version` (`model_id`, `provider_id`, `price_version`) USING BTREE,
    INDEX `idx_provider` (`provider_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI模型成本单价版本表'
  ROW_FORMAT = DYNAMIC;
