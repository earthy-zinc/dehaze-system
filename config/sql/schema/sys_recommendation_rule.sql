-- ============================================================
-- 表名: sys_recommendation_rule
-- 模块: 基础模块-推荐管理
-- ============================================================
-- 设计思路:
-- 推荐规则配置表，管理员可配置场景→算法的映射规则及权重。
-- scene_type 标识场景类型（urban/landscape/night/backlight/indoor 等）。
-- algorithm_ids 使用 JSON 数组存储该场景下的候选算法 ID 列表，避免拆关联表。
-- weight 控制规则优先级，数值越大越优先匹配。
-- enabled 控制规则启停，便于灰度发布和 A/B 测试。
-- 推荐规则使用逻辑删除，保留历史配置可追溯。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_recommendation_rule`;
CREATE TABLE `sys_recommendation_rule`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `rule_name`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '规则名称',
    `scene_type`   varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '场景类型(urban:城市;landscape:自然风景;night:夜景;backlight:逆光;indoor:室内)',
    `algorithm_ids` json                                                          NOT NULL COMMENT '候选算法ID列表（JSON数组）',
    `weight`       int                                                            NOT NULL DEFAULT 0 COMMENT '规则权重（数值越大越优先）',
    `enabled`      tinyint                                                        NOT NULL DEFAULT 1 COMMENT '是否启用(0:禁用;1:启用)',
    `deleted`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_scene_type` (`scene_type`) USING BTREE,
    INDEX `idx_enabled` (`enabled`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '推荐规则配置表'
  ROW_FORMAT = DYNAMIC;
