-- ============================================================
-- 表名: sys_knowledge_chunk_feedback
-- 模块: 基础模块-AI知识库
-- ============================================================
-- 设计思路:
-- AI 知识库分块片段级反馈表，记录用户对检索返回的某个 chunk 的点赞/点踩。
-- 低质量片段定义为"被用户点踩的 chunk"（thumbs_down_count = 该 chunk 被点踩次数）。
-- uk_chunk_user 唯一索引保证同一用户对同一片段只能反馈一次，再次反馈走 upsert 更新，
-- 天然幂等；一个 chunk 的点踩计数 = 对其 rating=-1 的记录条数（不去重，同用户只一条）。
-- rating 取值 1:点赞/-1:点踩；comment 存储可选的点踩原因。
-- 用户撤销反馈或点赞可覆盖原记录（rating 更新），点踩计数随之实时变化。
-- 分块记录本身只追加（见 sys_knowledge_chunk），反馈表随分块生命周期软删/保留均可，
-- 查询以 sys_knowledge_chunk 为纽带关联 kb，故不冗余 knowledge_base_id。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_knowledge_chunk_feedback`;
CREATE TABLE `sys_knowledge_chunk_feedback`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `chunk_id`        bigint                                                         NOT NULL COMMENT '分块ID(关联sys_knowledge_chunk.id)',
    `user_id`         bigint                                                         NOT NULL COMMENT '用户ID',
    `rating`          tinyint                                                        NOT NULL COMMENT '评分(1:点赞;-1:点踩)',
    `comment`         TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci        NULL COMMENT '反馈内容(可选,点踩原因)',
    `create_by`       bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`       bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`     datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_chunk_user` (`chunk_id`, `user_id`) USING BTREE,
    INDEX `idx_chunk` (`chunk_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI知识库分块反馈表'
  ROW_FORMAT = DYNAMIC;
