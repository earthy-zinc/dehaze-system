-- ============================================================
-- 表名: sys_favorite
-- 模块: 基础模块-收藏管理
-- ============================================================
-- 设计思路:
-- 统一收藏表，替代旧的 sys_algorithm_favorite 表，为所有可收藏对象提供通用收藏能力。
-- target_type + target_id 实现多态收藏，支持 algorithm/result/dataset 等任意业务实体。
-- uk_user_target 唯一索引防止同一用户重复收藏同一对象，同时支撑收藏状态批量查询。
-- idx_user_type_time 复合索引优化「按类型筛选 + 时间倒序」的收藏列表分页查询。
-- is_invalid 标识收藏对象是否已失效（对象被逻辑删除时置 1），列表中可一键清理。
-- 收藏记录使用逻辑删除，支持用户取消后重新收藏的历史追溯。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_favorite`;
CREATE TABLE `sys_favorite`
(
    `id`          bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`     bigint                                                         NOT NULL COMMENT '用户ID',
    `target_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '收藏对象类型(algorithm:算法;result:处理结果;dataset:数据集;image:图片;preset:参数方案)',
    `target_id`   bigint                                                         NOT NULL COMMENT '收藏对象ID',
    `is_invalid`  tinyint                                                        NOT NULL DEFAULT 0 COMMENT '收藏对象是否已失效(0:正常;1:已失效)',
    `deleted`     tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time` datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
    `update_time` datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`   bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`   bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_target` (`user_id`, `target_type`, `target_id`) USING BTREE,
    INDEX `idx_user_type_time` (`user_id`, `target_type`, `create_time` DESC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '统一收藏表'
  ROW_FORMAT = DYNAMIC;
