-- ============================================================
-- 表名: sys_algorithm
-- 模块: 业务模块
-- ============================================================
-- 设计思路:
-- 算法模型表，树形结构，parent_id 实现算法分类层级。
-- status 字段表示 6 种生命周期状态：1=草稿/2=测试中/3=待审核/4=已发布/5=已停用/6=已归档，
-- 与其他表的启用/禁用(0/1)二值状态含义不同。
-- path 字段存储模型文件在 nginx-dataset 中的相对路径，通过 HTTP HEAD 校验存在性。
-- audit_by/audit_time/audit_remark 记录审核信息，支撑算法发布审核流程。
-- recommend_score 由推荐管理模块基于用户评分、处理成功率、推荐采纳率综合计算并回写(0-5分)，
-- 作为推荐排序权重因子之一，新算法默认 3.5 分（冷启动策略）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_algorithm`;
CREATE TABLE `sys_algorithm`
(
    `id`           bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `parent_id`    bigint                                                         NULL DEFAULT 0 COMMENT '模型的父id',
    `type`         varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '模型类型',
    `version`      varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '算法版本号',
    `name`         varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '模型名称',
    `img`          TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL DEFAULT NULL COMMENT '模型图片',
    `path`         varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '模型存储路径',
    `size`         varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '模型大小',
    `params`       varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '模型参数',
    `flops`        varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '模型浮点运算次数',
    `import_path`  varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '模型代码导入路径',
    `description`  varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '针对该模型的详细描述',
    `status`       tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:草稿;2:测试中;3:待审核;4:已发布;5:已停用;6:已归档)',
    `audit_by`     bigint                                                         NULL DEFAULT NULL COMMENT '审核人ID',
    `audit_time`   datetime                                                       NULL DEFAULT NULL COMMENT '审核时间',
    `audit_remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '审核备注',
    `recommend_score` decimal(3,2)                                                NULL DEFAULT 3.50 COMMENT '推荐评分(0-5分，由推荐管理模块计算回写，新算法默认3.5)',
    `deleted`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '算法模型表'
  ROW_FORMAT = DYNAMIC;
