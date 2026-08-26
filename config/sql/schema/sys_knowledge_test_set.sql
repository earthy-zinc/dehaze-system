-- ============================================================
-- 表名: sys_knowledge_test_set
-- 模块: 基础模块-AI知识库
-- ============================================================
-- 设计思路:
-- 召回测试集：知识库管理页"召回测试"面板沉淀的评估基线（对齐《后端实现-检索引擎.md》§7.1）。
-- 单个测试集 = 一个用例（一条问题 + 期望命中分块），重复执行以对比调整分块/检索参数后的
-- Recall@K 与命中率。不拆分 item 表，避免过度设计。
-- expected_chunk_ids 存期望命中分块 ID 的 JSON 数组（must_include），与 sys_knowledge_chunk.id 对应。
-- 无业务唯一键（允许同一知识库下多条相似问题），软删除，按标准逻辑删除处理。
-- 路由层做知识库归属/权限校验（kb:manage），本表仅承载数据。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_knowledge_test_set`;
CREATE TABLE `sys_knowledge_test_set`
(
    `id`                 bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `knowledge_base_id`  bigint       NOT NULL COMMENT '知识库ID(关联sys_knowledge_base.id)',
    `question`           varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '测试问题',
    `expected_chunk_ids` json         NOT NULL COMMENT '期望命中分块ID数组(JSON，关联sys_knowledge_chunk.id)',
    `deleted`            tinyint      NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`          bigint       NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`          bigint       NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`        datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`        datetime     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_kb_id` (`knowledge_base_id`, `deleted`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI知识库召回测试集'
  ROW_FORMAT = DYNAMIC;
