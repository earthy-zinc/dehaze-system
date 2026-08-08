-- ============================================================
-- 表名: sys_knowledge_chunk
-- 模块: 基础模块-AI知识库
-- ============================================================
-- 设计思路:
-- AI 知识库分块表，记录文档分块后的向量化片段，是检索的基本单元。
- 一条文档分块后产生多条 chunk，每条 chunk 向量化后写入 ES 做向量检索。
- chunk_index 为分块序号（从 0 开始），用于引用溯源时展示来源位置。
- content 为分块后的文本片段，token_count 为该分块的 token 数。
- metadata(JSON) 存储分块元数据（来源文档/页码/段落/表格行等），检索时用于引用展示。
- embedding 可存储向量（备用于 MySQL 降级检索），但主要检索走 ES dense_vector。
- 分块记录为只追加（文档更新时删除旧分块重新分块），不使用逻辑删除。
- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_knowledge_chunk`;
CREATE TABLE `sys_knowledge_chunk`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `document_id`     bigint                                                         NOT NULL COMMENT '文档ID(关联sys_knowledge_document.id)',
    `knowledge_base_id` bigint                                                       NOT NULL COMMENT '知识库ID(冗余，便于跨文档检索)',
    `chunk_index`     int                                                            NOT NULL COMMENT '分块序号(从0开始)',
    `content`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NOT NULL COMMENT '分块后的文本片段',
    `token_count`     int                                                            NOT NULL DEFAULT 0 COMMENT '分块Token数',
    `metadata`        json                                                           NULL COMMENT '分块元数据(来源文档/页码/段落/表格行等，检索时用于引用展示)',
    `embedding`       json                                                           NULL COMMENT '向量嵌入(JSON数组，备用MySQL降级检索)',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_document` (`document_id`, `chunk_index`) USING BTREE,
    INDEX `idx_kb` (`knowledge_base_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI知识库分块表'
  ROW_FORMAT = DYNAMIC;
