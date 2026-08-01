-- ============================================================
-- 表名: sys_recommendation
-- 模块: 基础模块-推荐管理
-- ============================================================
-- 设计思路:
-- 推荐记录表，记录每次推荐请求的完整信息（图像特征、推荐算法、用户反馈）。
-- image_md5 用于关联图像特征分析缓存（recommend:feature:{imageMd5}），相同图片复用分析结果。
-- top_algorithms 使用 JSON 存储推荐算法列表（Top 3 算法 ID 及匹配度），避免拆子表。
-- analysis_result 使用 JSON 存储图像特征分析向量（7 维：雾霾浓度/场景类型/光照条件/复杂度/颜色分布/分辨率/噪声）。
-- feedback 记录用户对推荐结果的整体反馈(0:未反馈;1:有用;2:无用)，用于推荐效果度量。
-- adopted_algorithm_id 记录用户最终采纳的算法 ID（null 表示未采纳），关联推荐采纳率统计。
-- 推荐记录为只追加，不使用逻辑删除；过期数据通过定时任务物理清理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_recommendation`;
CREATE TABLE `sys_recommendation`
(
    `id`                  bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`             bigint                                                         NOT NULL COMMENT '用户ID',
    `image_md5`           char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci    NULL DEFAULT NULL COMMENT '图像MD5（关联特征分析缓存）',
    `target_type`         varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL DEFAULT 'algorithm' COMMENT '推荐对象类型(algorithm:算法;preset:参数方案;dataset:数据集;enhance:增强策略)',
    `top_algorithms`      json                                                           NOT NULL COMMENT '推荐算法列表（JSON数组，Top 3：算法ID及匹配度）',
    `analysis_result`     json                                                           NULL DEFAULT NULL COMMENT '图像特征分析结果（JSON：7维特征向量）',
    `feedback`            tinyint                                                        NOT NULL DEFAULT 0 COMMENT '推荐反馈(0:未反馈;1:有用;2:无用)',
    `adopted_algorithm_id` bigint                                                        NULL DEFAULT NULL COMMENT '用户采纳的算法ID（null:未采纳）',
    `create_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '推荐时间',
    `update_time`         datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`           bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`           bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_user_id` (`user_id`) USING BTREE,
    INDEX `idx_image_md5` (`image_md5`) USING BTREE,
    INDEX `idx_create_time` (`create_time`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '推荐记录表'
  ROW_FORMAT = DYNAMIC;
