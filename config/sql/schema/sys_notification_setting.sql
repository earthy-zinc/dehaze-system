-- ============================================================
-- 表名: sys_notification_setting
-- 模块: 消息通知
-- ============================================================
-- 设计思路:
-- 通知偏好设置表，每用户一行记录（user_id 唯一索引），存储用户的通知渠道偏好和免打扰设置。
-- push_enabled 控制 APP 推送总开关，站内信作为基本渠道不可关闭（由业务逻辑保证）。
-- dnd_enabled / dnd_start / dnd_end 管理免打扰时段，仅影响 APP 推送，不影响站内信。
-- preferences 字段(JSON) 存储细粒度偏好，包括按消息类型的推送开关和按业务模块的通知开关：
--   {"type_channels": {"announcement": {"push": true}, "business": {"push": false}},
--    "module_switches": {"prediction": true, "feedback": true, "announcement": true}}
-- 用户注册时自动初始化默认设置，用户修改后即时生效。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_notification_setting`;
CREATE TABLE `sys_notification_setting`
(
    `id`           bigint                                                        NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`      bigint                                                        NOT NULL COMMENT '用户ID',
    `push_enabled` tinyint                                                       NOT NULL DEFAULT 1 COMMENT 'APP推送总开关(1:开;0:关)',
    `dnd_enabled`  tinyint                                                       NOT NULL DEFAULT 0 COMMENT '免打扰开关(1:开;0:关)',
    `dnd_start`    time                                                          NULL DEFAULT '22:00:00' COMMENT '免打扰开始时间',
    `dnd_end`      time                                                          NULL DEFAULT '08:00:00' COMMENT '免打扰结束时间',
    `preferences`  json                                                          NULL DEFAULT NULL COMMENT '细粒度偏好(JSON，含按类型/模块的推送开关)',
    `deleted`      tinyint                                                       NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_time`  datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`  datetime                                                      NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `create_by`    bigint                                                        NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`    bigint                                                        NULL DEFAULT NULL COMMENT '修改人ID',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_user_id` (`user_id` ASC) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '通知偏好设置表'
  ROW_FORMAT = DYNAMIC;
