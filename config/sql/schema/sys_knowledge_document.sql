-- ============================================================
-- 表名: sys_knowledge_document
-- 模块: 基础模块-AI知识库
-- ============================================================
-- 设计思路:
-- AI 知识库文档表，记录上传到知识库的原始文档。
-- 一条文档记录对应一个上传的文件，文档被分块后产生多条 sys_knowledge_chunk 记录。
-- parsing_strategy 记录文档解析策略（auto:自动选择;ocr:OCR解析;text:纯文本提取;table:表格提取）。
-- file_id 关联 sys_file（上传的原始文件），通过 FileService 获取文件内容进行解析。
-- 解析后的纯文本存 content，原始富文本（含表格等）可存 raw_content。
-- chunk_count/total_tokens 冗余统计，便于展示处理进度。
-- processing_status 跟踪异步处理状态：待处理→处理中→已完成/失败。
-- 文档无业务唯一键（允许同名文档上传），软删除，按标准逻辑删除处理。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_knowledge_document`;
CREATE TABLE `sys_knowledge_document`
(
    `id`               bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `knowledge_base_id` bigint                                                         NOT NULL COMMENT '知识库ID(关联sys_knowledge_base.id)',
    `file_id`          bigint                                                          NULL DEFAULT NULL COMMENT '文件ID(关联sys_file.id，url导入与自定义文本无关联文件可为空)',
    `title`            varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文档标题(文件名或手动指定)',
    `source`           varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'upload' COMMENT '文档来源(manual:手动;upload:上传;url:URL导入;algorithm_sync:算法同步;experience:经验沉淀)',
    `version`          int                                                             NOT NULL DEFAULT 1 COMMENT '文档版本号(更新时+1，支撑版本回溯)',
    `parsing_strategy` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'auto' COMMENT '解析策略(auto:自动;ocr:OCR;text:纯文本;table:表格)',
    `content`          LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL COMMENT '解析后的纯文本内容',
    `raw_content`      LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci     NULL COMMENT '原始富文本(含Markdown/表格)',
    `chunk_count`      int                                                             NOT NULL DEFAULT 0 COMMENT '分块数(冗余统计)',
    `total_tokens`     bigint                                                          NOT NULL DEFAULT 0 COMMENT '编码Token总数(冗余统计)',
    `processing_status` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'pending' COMMENT '处理状态(pending:待处理;processing:处理中;completed:已完成;failed:失败)',
    `error`            TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NULL COMMENT '失败原因(processing_status=failed时填充)',
    `deleted`          tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`        bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`        bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`      datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`      datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_kb_status` (`knowledge_base_id`, `processing_status`, `deleted`) USING BTREE,
    INDEX `idx_file` (`file_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI知识库文档表'
  ROW_FORMAT = DYNAMIC;
