-- ============================================================
-- 表名: sys_message_template
-- 模块: 消息通知
-- ============================================================
-- 设计思路:
-- 消息模板表，存储各业务通知的标题和正文模板，支持变量占位符替换。
-- code 唯一索引，作为业务层引用模板的标识（如 member_level_up、order_refund_result）。
-- title_template 和 content_template 使用 {varName} 语法标记变量占位符，发送时由后端替换。
-- channels 字段(JSON)存储该模板的默认推送渠道配置，如 {"inbox": true, "push": true, "email": false}。
-- variables 字段(JSON)定义模板所需的变量名和说明，便于管理员维护模板时了解可用变量。
-- 管理员可在后台编辑模板内容，实现消息文案的动态调整而无需修改代码。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_message_template`;
CREATE TABLE `sys_message_template`
(
    `id`               bigint                                                         NOT NULL AUTO_INCREMENT COMMENT '主键',
    `code`             varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '模板编码(唯一标识，如member_level_up)',
    `name`             varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '模板名称',
    `type`             varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '消息类型(business:业务通知;member:会员通知;alert:告警通知)',
    `title_template`   varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT '' COMMENT '标题模板({varName}变量占位符)',
    `content_template` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci         NOT NULL COMMENT '正文模板({varName}变量占位符)',
    `priority`         tinyint                                                        NOT NULL DEFAULT 2 COMMENT '默认优先级(1:低;2:中;3:高;4:紧急)',
    `channels`         json                                                           NULL DEFAULT NULL COMMENT '默认推送渠道(JSON，如{"inbox":true,"push":true,"email":false})',
    `variables`        json                                                           NULL DEFAULT NULL COMMENT '变量定义(JSON，如[{"name":"levelName","desc":"等级名称"}])',
    `status`           tinyint                                                        NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`          tinyint                                                        NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`        bigint                                                         NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`        bigint                                                         NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`      datetime                                                       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_code` (`code` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '消息模板表'
  ROW_FORMAT = DYNAMIC;
