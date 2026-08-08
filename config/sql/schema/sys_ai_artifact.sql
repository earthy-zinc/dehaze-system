-- ============================================================
-- 表名: sys_ai_artifact
-- 模块: 核心模块-AI对话
-- ============================================================
-- 设计思路:
-- AI 中间产物表，实现需求规格中的 artifacts 引用化策略。
-- 图像处理结果、指标数据等大体积产物绝不塞入 LLM 对话上下文，只在 messages 中放 ID + 摘要元数据。
-- type 标识产物类型（image_result/metric_report/algorithm_recommend/file_ref）。
-- ref_type + ref_id 实现多态引用，关联 sys_pred_log/sys_eval_log/sys_file 等业务表。
-- summary(JSON) 只存储业务摘要（指标数值 + 算法信息 + 处理参数），绝不存储 URL。
--   图片 URL 通过 ref_type=sys_file + ref_id=file_id 引用 sys_file，运行时调 FileService.getUrl() 拼接，
--   遵循文件管理规范"URL 永远运行时拼接、永不落库"。
-- is_invalid 标识引用的业务对象是否已失效（如 sys_file 被删除时置 1），对齐 sys_favorite.is_invalid 设计。
-- State 中只存 artifact ID + 摘要，内容不进 messages，降低上下文 token 消耗。
-- 产物记录为只追加，不使用逻辑删除（关联的业务记录删除时，artifact 标记 is_invalid=1 保留引用痕迹）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_artifact`;
CREATE TABLE `sys_ai_artifact`
(
    `id`              bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `conversation_id` bigint                                                         NOT NULL COMMENT '会话ID',
    `message_id`      bigint                                                         NOT NULL COMMENT '关联消息ID(产生该产物的消息)',
    `type`            varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '产物类型(image_result:处理结果;metric_report:指标报告;algorithm_recommend:算法推荐;file_ref:文件引用)',
    `ref_type`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '引用业务表(sys_pred_log;sys_eval_log;sys_file等)',
    `ref_id`          bigint                                                         NULL DEFAULT NULL COMMENT '引用业务表ID',
    `summary`         json                                                           NULL COMMENT '业务摘要元数据(指标数值+算法信息+处理参数，绝不存URL)',
    `is_invalid`      tinyint                                                        NOT NULL DEFAULT 0 COMMENT '引用对象是否已失效(0:正常;1:已失效，如sys_file被删除时置1)',
    `create_time`     datetime                                                       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_conversation` (`conversation_id`) USING BTREE,
    INDEX `idx_message` (`message_id`) USING BTREE,
    INDEX `idx_ref` (`ref_type`, `ref_id`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = 'AI中间产物表'
  ROW_FORMAT = DYNAMIC;
