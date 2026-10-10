-- ============================================================
-- 表名: sys_ai_user_preference
-- 模块: AI 模型管理
-- ============================================================
-- 设计思路:
-- 用户级 AI 偏好设置表，每用户一行（user_id 唯一索引）。
-- default_model 存储用户偏好的默认对话模型（存 sys_ai_model.model_id，与 sys_ai_conversation.model 同口径）；
--   NULL 表示未设置（或已清除），此时按平台默认模型解析。
-- 默认模型解析链（请求指定 > 会话已存 > 用户偏好 > 平台默认，见 sys_dict ai_model.default_model）：
--   请求/会话均未指定模型时先取本表 default_model（该模型须存在、启用、model_type=chat），
--   取不到（未设置/模型已停用删除）回落平台默认模型；平台默认未配置则直接报错，不回落代码常量。
-- 本表仅承载"面向用户"的对话默认模型；后台派生推理（标题生成/记忆抽取/建议生成/试运行）不读本表。
-- 写入即生效（无需缓存失效，解析时每次读库）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_ai_user_preference`;
CREATE TABLE `sys_ai_user_preference`
(
    `id`            bigint                                                       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`       bigint                                                       NOT NULL COMMENT '用户ID',
    `default_model` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '偏好默认对话模型(model_id;空=未设置,回落平台默认模型)',
    `deleted`       bigint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)',
    `create_time`   datetime                                                     NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`     bigint                                                       NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                       NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_id` (`user_id` ASC, `deleted`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户AI偏好设置表'
  ROW_FORMAT = DYNAMIC;
