-- 2026-09-12 父子分块：sys_knowledge_chunk 加小节归属两列
-- 开发库增量执行（测试库由 conftest 全量重建自动生效）；
-- 存量 ES 分块索引需删除重建（section 元数据由 reprocess 重算）
ALTER TABLE `sys_knowledge_chunk`
    ADD COLUMN `section_index` int NOT NULL DEFAULT 0 COMMENT '所属小节序号(文档内递增,0=无标题文档整篇)' AFTER `chunk_index`,
    ADD COLUMN `section_path` varchar(255) NULL DEFAULT NULL COMMENT '小节标题路径(如 "4 核心设计 > 4.2 检索引擎",注入上下文作锚点)' AFTER `section_index`;
