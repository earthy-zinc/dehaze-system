-- ============================================================
-- 表名: sys_knowledge_base
-- 模块: 基础模块-AI知识库
-- ============================================================
-- 设计思路:
-- AI 知识库主表，管理知识库集合。一个知识库包含多个文档，一个文档包含多个分块。
-- 参考 Dify/RAGFlow 的知识库架构，区分知识库级别配置与文档级别管理。
-- embedding_provider/embedding_model 记录向量化策略（不同知识库可用不同 embedding 模型）。
-- chunking_strategy 记录分块策略（fixed:固定长度;semantic:语义切分;recursive:递归切分;qa:问答对;table:表格感知）。
-- search_strategy 记录检索策略（vector:纯向量;keyword:纯关键词;hybrid:混合检索）。
-- document_count/chunk_count/total_tokens 冗余统计，避免 JOIN 查询。
-- 知识库无业务唯一键（允许同名），软删除，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_knowledge_base`;
CREATE TABLE `sys_knowledge_base`
(
    `id`                 bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`               varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '知识库名称',
    `description`        TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '知识库描述',
    `visibility`         varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'private' COMMENT '可见性(public:平台公共库全员只读;private:私有库仅创建者可读写)',
    `embedding_provider` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'openai' COMMENT 'Embedding提供商(openai;qwen;cohere;local等)',
    `embedding_model`    varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'text-embedding-3-small' COMMENT 'Embedding模型标识(如text-embedding-3-small;bge-m3等)',
    `chunking_strategy`  varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'semantic' COMMENT '分块策略(fixed:固定长度;semantic:语义切分;recursive:递归切分;qa:问答对;table:表格感知)',
    `chunk_size`         int                                                             NOT NULL DEFAULT 800 COMMENT '分块大小(token数，范围的中间值)',
    `chunk_overlap`      int                                                             NOT NULL DEFAULT 80 COMMENT '分块重叠数(token)',
    `search_strategy`    varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'hybrid' COMMENT '检索策略(vector:纯向量;keyword:纯关键词BM25;hybrid:混合检索)',
    `hybrid_weight`      decimal(3,2)                                                    NOT NULL DEFAULT 0.70 COMMENT '混合检索中向量权重(0-1，剩余为关键词权重)',
    `top_k`              int                                                             NOT NULL DEFAULT 5 COMMENT '默认检索Top-K数',
    `score_threshold`    decimal(4,3)                                                    NOT NULL DEFAULT 0.500 COMMENT '相似度阈值(低于此分数的结果不返回)',
    `enable_rerank`      tinyint                                                         NOT NULL DEFAULT 0 COMMENT '是否启用重排序(0:否;1:是,需额外Rerank模型)',
    `rerank_model`       varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '重排序模型标识(bge-reranker-v2-m3等)',
    `document_count`     int                                                             NOT NULL DEFAULT 0 COMMENT '文档数(冗余统计)',
    `chunk_count`        int                                                             NOT NULL DEFAULT 0 COMMENT '分块总数(冗余统计)',
    `total_tokens`       bigint                                                          NOT NULL DEFAULT 0 COMMENT '编码Token总数(冗余统计)',
    `status`             tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;2:处理中;0:禁用)',
    `deleted`            tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`          bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`          bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`        datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`        datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_status` (`status`, `deleted`) USING BTREE,
    INDEX `idx_create_by_visibility` (`create_by`, `visibility`, `deleted`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI知识库主表'
  ROW_FORMAT = DYNAMIC;
