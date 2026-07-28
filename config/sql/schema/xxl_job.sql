-- ============================================================
-- XXL-JOB 调度中心库
-- 来源: xxl-job 3.3.0 官方 tables_xxl_job.sql
-- 改造: 删库重建、清除示例数据、配置项目实际执行器与任务
-- ============================================================
-- 设计思路:
-- xxl_job 库独立于业务库 dehaze，由 xxl-job-admin 容器独立访问。
-- 预置 3 个执行器分组（dehaze-java/python/go）和项目中实际存在的定时任务，
-- 任务初始为停止状态（trigger_status=0），admin 启动后在控制台手动启用。
-- admin 控制台账号 admin/12345678（与项目其他服务密码一致）。
-- ------------------------------------------------------------

DROP DATABASE IF EXISTS `xxl_job`;
CREATE DATABASE `xxl_job` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `xxl_job`;

SET NAMES utf8mb4;

-- ------------------------------------------------------------
-- 表结构（与官方 3.3.0 一致，不修改）
-- ------------------------------------------------------------

CREATE TABLE `xxl_job_group`
(
    `id`           int(11)     NOT NULL AUTO_INCREMENT,
    `app_name`     varchar(64) NOT NULL COMMENT '执行器AppName',
    `title`        varchar(12) NOT NULL COMMENT '执行器名称',
    `address_type` tinyint(4)  NOT NULL DEFAULT '0' COMMENT '执行器地址类型：0=自动注册、1=手动录入',
    `address_list` text COMMENT '执行器地址列表，多地址逗号分隔',
    `update_time`  datetime             DEFAULT NULL,
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_registry`
(
    `id`             int(11)      NOT NULL AUTO_INCREMENT,
    `registry_group` varchar(50)  NOT NULL,
    `registry_key`   varchar(255) NOT NULL,
    `registry_value` varchar(255) NOT NULL,
    `update_time`    datetime DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `i_g_k_v` (`registry_group`, `registry_key`, `registry_value`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_info`
(
    `id`                        int(11)      NOT NULL AUTO_INCREMENT,
    `job_group`                 int(11)      NOT NULL COMMENT '执行器主键ID',
    `job_desc`                  varchar(255) NOT NULL,
    `add_time`                  datetime              DEFAULT NULL,
    `update_time`               datetime              DEFAULT NULL,
    `author`                    varchar(64)           DEFAULT NULL COMMENT '作者',
    `alarm_email`               varchar(255)          DEFAULT NULL COMMENT '报警邮件',
    `schedule_type`             varchar(50)  NOT NULL DEFAULT 'NONE' COMMENT '调度类型',
    `schedule_conf`             varchar(128)          DEFAULT NULL COMMENT '调度配置，值含义取决于调度类型',
    `misfire_strategy`          varchar(50)  NOT NULL DEFAULT 'DO_NOTHING' COMMENT '调度过期策略',
    `executor_route_strategy`   varchar(50)           DEFAULT NULL COMMENT '执行器路由策略',
    `executor_handler`          varchar(255)          DEFAULT NULL COMMENT '执行器任务handler',
    `executor_param`            varchar(512)          DEFAULT NULL COMMENT '执行器任务参数',
    `executor_block_strategy`   varchar(50)           DEFAULT NULL COMMENT '阻塞处理策略',
    `executor_timeout`          int(11)      NOT NULL DEFAULT '0' COMMENT '任务执行超时时间，单位秒',
    `executor_fail_retry_count` int(11)      NOT NULL DEFAULT '0' COMMENT '失败重试次数',
    `glue_type`                 varchar(50)  NOT NULL COMMENT 'GLUE类型',
    `glue_source`               mediumtext COMMENT 'GLUE源代码',
    `glue_remark`               varchar(128)          DEFAULT NULL COMMENT 'GLUE备注',
    `glue_updatetime`           datetime              DEFAULT NULL COMMENT 'GLUE更新时间',
    `child_jobid`               varchar(255)          DEFAULT NULL COMMENT '子任务ID，多个逗号分隔',
    `trigger_status`            tinyint(4)   NOT NULL DEFAULT '0' COMMENT '调度状态：0-停止，1-运行',
    `trigger_last_time`         bigint(13)   NOT NULL DEFAULT '0' COMMENT '上次调度时间',
    `trigger_next_time`         bigint(13)   NOT NULL DEFAULT '0' COMMENT '下次调度时间',
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_logglue`
(
    `id`          int(11)      NOT NULL AUTO_INCREMENT,
    `job_id`      int(11)      NOT NULL COMMENT '任务，主键ID',
    `glue_type`   varchar(50) DEFAULT NULL COMMENT 'GLUE类型',
    `glue_source` mediumtext COMMENT 'GLUE源代码',
    `glue_remark` varchar(128) NOT NULL COMMENT 'GLUE备注',
    `add_time`    datetime    DEFAULT NULL,
    `update_time` datetime    DEFAULT NULL,
    PRIMARY KEY (`id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_log`
(
    `id`                        bigint(20) NOT NULL AUTO_INCREMENT,
    `job_group`                 int(11)    NOT NULL COMMENT '执行器主键ID',
    `job_id`                    int(11)    NOT NULL COMMENT '任务，主键ID',
    `executor_address`          varchar(255)        DEFAULT NULL COMMENT '执行器地址，本次执行的地址',
    `executor_handler`          varchar(255)        DEFAULT NULL COMMENT '执行器任务handler',
    `executor_param`            varchar(512)        DEFAULT NULL COMMENT '执行器任务参数',
    `executor_sharding_param`   varchar(20)         DEFAULT NULL COMMENT '执行器任务分片参数，格式如 1/2',
    `executor_fail_retry_count` int(11)    NOT NULL DEFAULT '0' COMMENT '失败重试次数',
    `trigger_time`              datetime            DEFAULT NULL COMMENT '调度-时间',
    `trigger_code`              int(11)    NOT NULL COMMENT '调度-结果',
    `trigger_msg`               text COMMENT '调度-日志',
    `handle_time`               datetime            DEFAULT NULL COMMENT '执行-时间',
    `handle_code`               int(11)    NOT NULL COMMENT '执行-状态',
    `handle_msg`                text COMMENT '执行-日志',
    `alarm_status`              tinyint(4) NOT NULL DEFAULT '0' COMMENT '告警状态：0-默认、1-无需告警、2-告警成功、3-告警失败',
    PRIMARY KEY (`id`),
    KEY `I_trigger_time` (`trigger_time`),
    KEY `I_handle_code` (`handle_code`),
    KEY `I_jobid_jobgroup` (`job_id`, `job_group`),
    KEY `I_job_id` (`job_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_log_report`
(
    `id`            int(11) NOT NULL AUTO_INCREMENT,
    `trigger_day`   datetime         DEFAULT NULL COMMENT '调度-时间',
    `running_count` int(11) NOT NULL DEFAULT '0' COMMENT '运行中-日志数量',
    `suc_count`     int(11) NOT NULL DEFAULT '0' COMMENT '执行成功-日志数量',
    `fail_count`    int(11) NOT NULL DEFAULT '0' COMMENT '执行失败-日志数量',
    `update_time`   datetime         DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `i_trigger_day` (`trigger_day`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_lock`
(
    `lock_name` varchar(50) NOT NULL COMMENT '锁名称',
    PRIMARY KEY (`lock_name`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

CREATE TABLE `xxl_job_user`
(
    `id`         int(11)      NOT NULL AUTO_INCREMENT,
    `username`   varchar(50)  NOT NULL COMMENT '账号',
    `password`   varchar(100) NOT NULL COMMENT '密码加密信息',
    `token`      varchar(100) DEFAULT NULL COMMENT '登录token',
    `role`       tinyint(4)   NOT NULL COMMENT '角色：0-普通用户、1-管理员',
    `permission` varchar(255) DEFAULT NULL COMMENT '权限：执行器ID列表，多个逗号分割',
    PRIMARY KEY (`id`),
    UNIQUE KEY `i_username` (`username`) USING BTREE
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- ------------------------------------------------------------
-- 初始数据
-- ------------------------------------------------------------

-- 执行器分组（3 个后端各一个）
INSERT INTO `xxl_job_group`(`id`, `app_name`, `title`, `address_type`, `address_list`, `update_time`)
VALUES (1, 'xxl-job-executor-dehaze-java', 'Java后端执行器', 0, NULL, now()),
       (2, 'xxl-job-executor-dehaze-python', 'Python后端执行器', 0, NULL, now()),
       (3, 'xxl-job-executor-dehaze-go', 'Go后端执行器', 0, NULL, now());

-- 调度锁
INSERT INTO `xxl_job_lock` (`lock_name`)
VALUES ('schedule_lock');

-- 控制台用户：admin / 12345678（SHA-256）
INSERT INTO `xxl_job_user`(`id`, `username`, `password`, `role`, `permission`)
VALUES (1, 'admin', 'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f', 1, NULL);

-- ------------------------------------------------------------
-- 定时任务（三端对齐后的完整清单）
-- schedule_type=CRON, glue_type=BEAN, trigger_status=0(停止)
-- 路由策略: SHARDING_BROADCAST 用于可分片任务, FIRST 用于单机任务
-- 阻塞策略: SERIAL_EXECUTION 串行执行
-- ------------------------------------------------------------

-- === Java 执行器（job_group=1）任务 ===
INSERT INTO `xxl_job_info`(`job_group`, `job_desc`, `add_time`, `update_time`, `author`, `alarm_email`,
                           `schedule_type`, `schedule_conf`, `misfire_strategy`, `executor_route_strategy`,
                           `executor_handler`, `executor_param`, `executor_block_strategy`, `executor_timeout`,
                           `executor_fail_retry_count`, `glue_type`, `glue_source`, `glue_remark`, `glue_updatetime`,
                           `child_jobid`, `trigger_status`)
VALUES
    -- 任务清理-过期任务物理删除
    (1, '任务清理-过期任务物理删除', now(), now(), 'dehaze', '', 'CRON', '0 0 2 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 任务清理-僵死任务标记失败
    (1, '任务清理-僵死任务标记失败', now(), now(), 'dehaze', '', 'CRON', '0 0 * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 预测评估日志-僵尸任务恢复
    (1, '预测评估日志-僵尸任务恢复', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckPredEvalLogs', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-超时未支付自动取消
    (1, '订单-超时未支付自动取消', now(), now(), 'dehaze', '', 'CRON', '0 */5 * * * ?',
     'DO_NOTHING', 'FIRST', 'expireOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-套餐到期自动完成
    (1, '订单-套餐到期自动完成', now(), now(), 'dehaze', '', 'CRON', '0 0 3 * * ?',
     'DO_NOTHING', 'FIRST', 'completeExpiredOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-自动续费
    (1, '订单-自动续费', now(), now(), 'dehaze', '', 'CRON', '0 0 3 * * ?',
     'DO_NOTHING', 'FIRST', 'autoRenew', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 优惠券-用户优惠券过期处理
    (1, '优惠券-用户优惠券过期处理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'expireUserCoupons', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 会员-过期降级
    (1, '会员-过期降级', now(), now(), 'dehaze', '', 'CRON', '0 0 2 * * ?',
     'DO_NOTHING', 'FIRST', 'processExpiredMembers', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 会员-月度配额重置
    (1, '会员-月度配额重置', now(), now(), 'dehaze', '', 'CRON', '0 0 0 1 * ?',
     'DO_NOTHING', 'FIRST', 'resetMonthlyQuota', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 公告-定时发送
    (1, '公告-定时发送', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'sendScheduledAnnouncements', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 消息-过期消息清理
    (1, '消息-过期消息清理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredMessages', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0);

-- === Python 执行器（job_group=2）任务 ===
INSERT INTO `xxl_job_info`(`job_group`, `job_desc`, `add_time`, `update_time`, `author`, `alarm_email`,
                           `schedule_type`, `schedule_conf`, `misfire_strategy`, `executor_route_strategy`,
                           `executor_handler`, `executor_param`, `executor_block_strategy`, `executor_timeout`,
                           `executor_fail_retry_count`, `glue_type`, `glue_source`, `glue_remark`, `glue_updatetime`,
                           `child_jobid`, `trigger_status`)
VALUES
    -- 任务清理-过期任务物理删除
    (2, '任务清理-过期任务物理删除', now(), now(), 'dehaze', '', 'CRON', '0 0 2 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 任务清理-僵死任务标记失败
    (2, '任务清理-僵死任务标记失败', now(), now(), 'dehaze', '', 'CRON', '0 0 * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 预测评估日志-僵尸任务恢复
    (2, '预测评估日志-僵尸任务恢复', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckPredEvalLogs', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-超时未支付自动取消
    (2, '订单-超时未支付自动取消', now(), now(), 'dehaze', '', 'CRON', '0 */5 * * * ?',
     'DO_NOTHING', 'FIRST', 'expireOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-套餐到期自动完成
    (2, '订单-套餐到期自动完成', now(), now(), 'dehaze', '', 'CRON', '0 0 3 * * ?',
     'DO_NOTHING', 'FIRST', 'completeExpiredOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 优惠券-用户优惠券过期处理
    (2, '优惠券-用户优惠券过期处理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'expireUserCoupons', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 公告-定时发送
    (2, '公告-定时发送', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'sendScheduledAnnouncements', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 消息-过期消息清理
    (2, '消息-过期消息清理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredMessages', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 文件清理-孤儿文件清理（Python特有，算法端产生孤儿文件）
    (2, '文件清理-孤儿文件清理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupOrphanFiles', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 文件清理-临时文件清理（Python特有，算法端产生临时文件）
    (2, '文件清理-临时文件清理', now(), now(), 'dehaze', '', 'CRON', '0 0 */6 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupTempFiles', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 模型健康检查（Python特有，GPU/DB/Redis健康监控）
    (2, '模型健康检查', now(), now(), 'dehaze', '', 'CRON', '0 */30 * * * ?',
     'DO_NOTHING', 'FIRST', 'modelHealthCheck', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0);

-- === Go 执行器（job_group=3）任务 ===
INSERT INTO `xxl_job_info`(`job_group`, `job_desc`, `add_time`, `update_time`, `author`, `alarm_email`,
                           `schedule_type`, `schedule_conf`, `misfire_strategy`, `executor_route_strategy`,
                           `executor_handler`, `executor_param`, `executor_block_strategy`, `executor_timeout`,
                           `executor_fail_retry_count`, `glue_type`, `glue_source`, `glue_remark`, `glue_updatetime`,
                           `child_jobid`, `trigger_status`)
VALUES
    -- 任务清理-过期任务物理删除
    (3, '任务清理-过期任务物理删除', now(), now(), 'dehaze', '', 'CRON', '0 0 2 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 任务清理-僵死任务标记失败
    (3, '任务清理-僵死任务标记失败', now(), now(), 'dehaze', '', 'CRON', '0 0 * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckTasks', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 预测评估日志-僵尸任务恢复
    (3, '预测评估日志-僵尸任务恢复', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupStuckPredEvalLogs', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-超时未支付自动取消
    (3, '订单-超时未支付自动取消', now(), now(), 'dehaze', '', 'CRON', '0 */5 * * * ?',
     'DO_NOTHING', 'FIRST', 'expireOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-套餐到期自动完成
    (3, '订单-套餐到期自动完成', now(), now(), 'dehaze', '', 'CRON', '0 0 3 * * ?',
     'DO_NOTHING', 'FIRST', 'completeExpiredOrders', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 订单-自动续费
    (3, '订单-自动续费', now(), now(), 'dehaze', '', 'CRON', '0 0 3 * * ?',
     'DO_NOTHING', 'FIRST', 'autoRenew', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 优惠券-用户优惠券过期处理
    (3, '优惠券-用户优惠券过期处理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'expireUserCoupons', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 公告-定时发送
    (3, '公告-定时发送', now(), now(), 'dehaze', '', 'CRON', '0 * * * * ?',
     'DO_NOTHING', 'FIRST', 'sendScheduledAnnouncements', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0),
    -- 消息-过期消息清理
    (3, '消息-过期消息清理', now(), now(), 'dehaze', '', 'CRON', '0 0 4 * * ?',
     'DO_NOTHING', 'FIRST', 'cleanupExpiredMessages', '', 'SERIAL_EXECUTION', 0, 0, 'BEAN', '', '', NULL, '', 0);

COMMIT;
