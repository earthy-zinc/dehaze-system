CREATE DATABASE IF NOT EXISTS `pei_iot`;
USE `pei_iot`;

-- ----------------------------
-- Table structure for iot_device
-- ----------------------------
DROP TABLE IF EXISTS `iot_device`;
CREATE TABLE `iot_device`
(
    `id`             bigint         NOT NULL AUTO_INCREMENT COMMENT '设备 ID，主键，自增',
    `device_key`     varchar(255)   NOT NULL DEFAULT '' COMMENT '设备唯一标识符，全局唯一，用于识别设备',
    `device_name`    varchar(255)   NOT NULL DEFAULT '' COMMENT '设备名称，在产品内唯一，用于标识设备',
    `nickname`       varchar(255)   NOT NULL DEFAULT '' COMMENT '设备备注名称',
    `serial_number`  varchar(255)   NOT NULL DEFAULT '' COMMENT '设备序列号',
    `pic_url`        varchar(1024)  NOT NULL DEFAULT '' COMMENT '设备图片',
    `group_ids`      text           NULL COMMENT '设备分组编号集合',
    `product_id`     bigint         NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_key`    varchar(255)   NOT NULL DEFAULT '' COMMENT '产品标识',
    `device_type`    int            NOT NULL DEFAULT 0 COMMENT '设备类型',
    `gateway_id`     bigint         NOT NULL DEFAULT 0 COMMENT '网关设备编号',
    `state`          int            NOT NULL DEFAULT 0 COMMENT '设备状态 (IotDeviceStateEnum)',
    `online_time`    datetime       NULL     DEFAULT NULL COMMENT '最后上线时间',
    `offline_time`   datetime       NULL     DEFAULT NULL COMMENT '最后离线时间',
    `active_time`    datetime       NULL     DEFAULT NULL COMMENT '设备激活时间',
    `ip`             varchar(64)    NOT NULL DEFAULT '' COMMENT '设备的 IP 地址',
    `firmware_id`    varchar(255)   NOT NULL DEFAULT '' COMMENT '固件编号',
    `device_secret`  varchar(255)   NOT NULL DEFAULT '' COMMENT '设备密钥，用于设备认证，需安全存储',
    `mqtt_client_id` varchar(255)   NOT NULL DEFAULT '' COMMENT 'MQTT 客户端 ID',
    `mqtt_username`  varchar(255)   NOT NULL DEFAULT '' COMMENT 'MQTT 用户名',
    `mqtt_password`  varchar(255)   NOT NULL DEFAULT '' COMMENT 'MQTT 密码',
    `auth_type`      varchar(255)   NOT NULL DEFAULT '' COMMENT '认证类型（如一机一密、动态注册）',
    `latitude`       decimal(10, 6) NULL     DEFAULT NULL COMMENT '设备位置的纬度',
    `longitude`      decimal(10, 6) NULL     DEFAULT NULL COMMENT '设备位置的经度',
    `area_id`        int            NOT NULL DEFAULT 0 COMMENT '地区编码',
    `address`        varchar(1024)  NOT NULL DEFAULT '' COMMENT '设备详细地址',
    `config`         text           NULL COMMENT '设备配置',
    `creator`        varchar(64)    NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)    NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)         NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`      bigint         NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_create_time` (`create_time` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 设备表';

-- ----------------------------
-- Table structure for iot_device_group
-- ----------------------------
DROP TABLE IF EXISTS `iot_device_group`;
CREATE TABLE `iot_device_group`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '分组 ID',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '分组名字',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '分组状态 (CommonStatusEnum)',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '分组描述',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE,
    INDEX `idx_create_time` (`create_time` ASC) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 设备分组表';

-- ----------------------------
-- Table structure for iot_ota_firmware
-- ----------------------------
DROP TABLE IF EXISTS `iot_ota_firmware`;
CREATE TABLE `iot_ota_firmware`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '固件编号',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '固件名称',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '固件版本',
    `version`     varchar(255)  NOT NULL DEFAULT '' COMMENT '版本号',
    `product_id`  bigint        NOT NULL DEFAULT '' COMMENT '产品编号',
    `product_key` varchar(255)  NOT NULL DEFAULT '' COMMENT '产品标识',
    `sign_method` varchar(64)   NOT NULL DEFAULT '' COMMENT '签名方式',
    `file_sign`   varchar(255)  NOT NULL DEFAULT '' COMMENT '固件文件签名',
    `file_size`   bigint        NOT NULL DEFAULT 0 COMMENT '固件文件大小',
    `file_url`    varchar(1024) NOT NULL DEFAULT '' COMMENT '固件文件 URL',
    `information` text          NULL COMMENT '自定义信息，建议使用 JSON 格式',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT OTA 固件表';

-- ----------------------------
-- Table structure for iot_ota_upgrade_record
-- ----------------------------
DROP TABLE IF EXISTS `iot_ota_upgrade_record`;
CREATE TABLE `iot_ota_upgrade_record`
(
    `id`               bigint        NOT NULL AUTO_INCREMENT COMMENT '升级记录编号',
    `firmware_id`      bigint        NOT NULL DEFAULT 0 COMMENT '固件编号',
    `task_id`          bigint        NOT NULL DEFAULT 0 COMMENT '任务编号',
    `product_key`      varchar(255)  NOT NULL DEFAULT '' COMMENT '产品标识',
    `device_name`      varchar(255)  NOT NULL DEFAULT '' COMMENT '设备名称',
    `device_id`        varchar(255)  NOT NULL DEFAULT '' COMMENT '设备编号',
    `from_firmware_id` bigint        NOT NULL DEFAULT 0 COMMENT '来源的固件编号',
    `status`           int           NOT NULL DEFAULT 0 COMMENT '升级状态 (IotOtaUpgradeRecordStatusEnum)',
    `progress`         int           NOT NULL DEFAULT 0 COMMENT '升级进度，百分比',
    `description`      varchar(1024) NOT NULL DEFAULT '' COMMENT '升级进度描述',
    `start_time`       datetime      NULL     DEFAULT NULL COMMENT '升级开始时间',
    `end_time`         datetime      NULL     DEFAULT NULL COMMENT '升级结束时间',
    `creator`          varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT OTA 升级记录表';

-- ----------------------------
-- Table structure for iot_ota_upgrade_task
-- ----------------------------
DROP TABLE IF EXISTS `iot_ota_upgrade_task`;
CREATE TABLE `iot_ota_upgrade_task`
(
    `id`           bigint        NOT NULL AUTO_INCREMENT COMMENT '任务编号',
    `name`         varchar(255)  NOT NULL DEFAULT '' COMMENT '任务名称',
    `description`  varchar(1024) NOT NULL DEFAULT '' COMMENT '任务描述',
    `firmware_id`  bigint        NOT NULL DEFAULT 0 COMMENT '固件编号',
    `status`       int           NOT NULL DEFAULT 0 COMMENT '任务状态 (IotOtaUpgradeTaskStatusEnum)',
    `scope`        int           NOT NULL DEFAULT 0 COMMENT '升级范围 (IotOtaUpgradeTaskScopeEnum)',
    `device_count` bigint        NOT NULL DEFAULT 0 COMMENT '设备数量',
    `device_ids`   text          NULL COMMENT '选中的设备编号数组',
    `creator`      varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`      varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`  datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`      bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT OTA 升级任务表';

-- ----------------------------
-- Table structure for iot_plugin_config
-- ----------------------------
DROP TABLE IF EXISTS `iot_plugin_config`;
CREATE TABLE `iot_plugin_config`
(
    `id`            bigint        NOT NULL AUTO_INCREMENT COMMENT '主键 ID',
    `plugin_key`    varchar(255)  NOT NULL DEFAULT '' COMMENT '插件包标识符',
    `name`          varchar(255)  NOT NULL DEFAULT '' COMMENT '插件名称',
    `description`   varchar(1024) NOT NULL DEFAULT '' COMMENT '插件描述',
    `deploy_type`   int           NOT NULL DEFAULT 0 COMMENT '部署方式 (IotPluginDeployTypeEnum)',
    `file_name`     varchar(255)  NOT NULL DEFAULT '' COMMENT '插件包文件名',
    `version`       varchar(255)  NOT NULL DEFAULT '' COMMENT '插件版本',
    `type`          int           NOT NULL DEFAULT 0 COMMENT '插件类型 (IotPluginTypeEnum)',
    `protocol`      varchar(255)  NOT NULL DEFAULT '' COMMENT '设备插件协议类型',
    `status`        int           NOT NULL DEFAULT 0 COMMENT '状态 (CommonStatusEnum)',
    `config_schema` text          NULL COMMENT '插件配置项描述信息',
    `config`        text          NULL COMMENT '插件配置信息',
    `script`        text          NULL COMMENT '插件脚本',
    `creator`       varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 插件配置表';

-- ----------------------------
-- Table structure for iot_plugin_instance
-- ----------------------------
DROP TABLE IF EXISTS `iot_plugin_instance`;
CREATE TABLE `iot_plugin_instance`
(
    `id`              bigint       NOT NULL AUTO_INCREMENT COMMENT '主键',
    `plugin_id`       bigint       NOT NULL DEFAULT 0 COMMENT '插件编号',
    `process_id`      varchar(255) NOT NULL DEFAULT '' COMMENT '插件进程编号',
    `host_ip`         varchar(64)  NOT NULL DEFAULT '' COMMENT '插件实例所在 IP',
    `downstream_port` int          NOT NULL DEFAULT 0 COMMENT '设备下行端口',
    `online`          bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否在线',
    `online_time`     datetime     NULL     DEFAULT NULL COMMENT '在线时间',
    `offline_time`    datetime     NULL     DEFAULT NULL COMMENT '离线时间',
    `heartbeat_time`  datetime     NULL     DEFAULT NULL COMMENT '心跳时间',
    `creator`         varchar(64)  NULL     DEFAULT '' COMMENT '创建者',
    `create_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`         varchar(64)  NULL     DEFAULT '' COMMENT '更新者',
    `update_time`     datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`         bit(1)       NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`       bigint       NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 插件实例表';

-- ----------------------------
-- Table structure for iot_product_category
-- ----------------------------
DROP TABLE IF EXISTS `iot_product_category`;
CREATE TABLE `iot_product_category`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '分类 ID',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '分类名字',
    `sort`        int           NOT NULL DEFAULT 0 COMMENT '分类排序',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '分类状态 (CommonStatusEnum)',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '分类描述',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 产品分类表';

-- ----------------------------
-- Table structure for iot_product
-- ----------------------------
DROP TABLE IF EXISTS `iot_product`;
CREATE TABLE `iot_product`
(
    `id`            bigint        NOT NULL AUTO_INCREMENT COMMENT '产品 ID',
    `name`          varchar(255)  NOT NULL DEFAULT '' COMMENT '产品名称',
    `product_key`   varchar(255)  NOT NULL DEFAULT '' COMMENT '产品标识',
    `category_id`   bigint        NOT NULL DEFAULT 0 COMMENT '产品分类编号',
    `icon`          varchar(1024) NOT NULL DEFAULT '' COMMENT '产品图标',
    `pic_url`       varchar(1024) NOT NULL DEFAULT '' COMMENT '产品图片',
    `description`   varchar(1024) NOT NULL DEFAULT '' COMMENT '产品描述',
    `status`        int           NOT NULL DEFAULT 0 COMMENT '产品状态 (IotProductStatusEnum)',
    `device_type`   int           NOT NULL DEFAULT 0 COMMENT '设备类型 (IotProductDeviceTypeEnum)',
    `net_type`      int           NOT NULL DEFAULT 0 COMMENT '联网方式 (IotNetTypeEnum)',
    `protocol_type` int           NOT NULL DEFAULT 0 COMMENT '接入网关协议 (IotProtocolTypeEnum)',
    `protocol_id`   bigint        NOT NULL DEFAULT 0 COMMENT '协议编号',
    `data_format`   int           NOT NULL DEFAULT 0 COMMENT '数据格式 (IotDataFormatEnum)',
    `validate_type` int           NOT NULL DEFAULT 0 COMMENT '数据校验级别 (IotValidateTypeEnum)',
    `creator`       varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`   datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`       varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`   datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`       bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`     bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 产品表';

-- ----------------------------
-- Table structure for iot_alert_config
-- ----------------------------
DROP TABLE IF EXISTS `iot_alert_config`;
CREATE TABLE `iot_alert_config`
(
    `id`               bigint        NOT NULL AUTO_INCREMENT COMMENT '配置编号',
    `name`             varchar(255)  NOT NULL DEFAULT '' COMMENT '配置名称',
    `description`      varchar(1024) NOT NULL DEFAULT '' COMMENT '配置描述',
    `level`            int           NOT NULL DEFAULT 0 COMMENT '配置状态',
    `status`           int           NOT NULL DEFAULT 0 COMMENT '配置状态 (CommonStatusEnum)',
    `rule_scene_ids`   text          NULL COMMENT '关联的规则场景编号数组',
    `receive_user_ids` text          NULL COMMENT '接收的用户编号数组',
    `receive_types`    text          NULL COMMENT '接收的类型数组 (IotAlertConfigReceiveTypeEnum)',
    `creator`          varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`          varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`      datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`          bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 告警配置表';

-- ----------------------------
-- Table structure for iot_alert_record
-- ----------------------------
DROP TABLE IF EXISTS `iot_alert_record`;
CREATE TABLE `iot_alert_record`
(
    `id`             bigint        NOT NULL AUTO_INCREMENT COMMENT '记录编号',
    `config_id`      bigint        NOT NULL DEFAULT 0 COMMENT '告警名称',
    `name`           varchar(255)  NOT NULL DEFAULT '' COMMENT '告警名称',
    `product_key`    varchar(255)  NOT NULL DEFAULT '' COMMENT '产品标识',
    `device_name`    varchar(255)  NOT NULL DEFAULT '' COMMENT '设备名称',
    `device_message` text          NULL COMMENT '触发的设备消息',
    `process_status` bit(1)        NOT NULL DEFAULT b'0' COMMENT '处理状态 true - 已处理 false - 未处理',
    `process_remark` varchar(1024) NOT NULL DEFAULT '' COMMENT '处理结果（备注）',
    `creator`        varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time`    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`        varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time`    datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`        bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 告警记录表';

-- ----------------------------
-- Table structure for iot_data_bridge
-- ----------------------------
DROP TABLE IF EXISTS `iot_data_bridge`;
CREATE TABLE `iot_data_bridge`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '桥梁编号',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '桥梁名称',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '桥梁描述',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '桥梁状态 (CommonStatusEnum)',
    `direction`   int           NOT NULL DEFAULT 0 COMMENT '桥梁方向 (IotDataBridgeDirectionEnum)',
    `type`        int           NOT NULL DEFAULT 0 COMMENT '桥梁类型 (IotDataBridgeTypeEnum)',
    `config`      text          NULL COMMENT '桥梁配置',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 数据桥梁表';

-- ----------------------------
-- Table structure for iot_rule_scene
-- ----------------------------
DROP TABLE IF EXISTS `iot_rule_scene`;
CREATE TABLE `iot_rule_scene`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '场景编号',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '场景名称',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '场景描述',
    `status`      int           NOT NULL DEFAULT 0 COMMENT '场景状态 (CommonStatusEnum)',
    `triggers`    text          NULL COMMENT '触发器数组',
    `actions`     text          NULL COMMENT '执行器数组',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    `tenant_id`   bigint        NOT NULL DEFAULT 0 COMMENT '租户编号',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 规则场景（场景联动）表';

-- ----------------------------
-- Table structure for iot_thing_model
-- ----------------------------
DROP TABLE IF EXISTS `iot_thing_model`;
CREATE TABLE `iot_thing_model`
(
    `id`          bigint        NOT NULL AUTO_INCREMENT COMMENT '物模型功能编号',
    `identifier`  varchar(255)  NOT NULL DEFAULT '' COMMENT '功能标识',
    `name`        varchar(255)  NOT NULL DEFAULT '' COMMENT '功能名称',
    `description` varchar(1024) NOT NULL DEFAULT '' COMMENT '功能描述',
    `product_id`  bigint        NOT NULL DEFAULT 0 COMMENT '产品编号',
    `product_key` varchar(255)  NOT NULL DEFAULT '' COMMENT '产品标识',
    `type`        int           NOT NULL DEFAULT 0 COMMENT '功能类型 (IotThingModelTypeEnum)',
    `property`    text          NULL COMMENT '属性',
    `event`       text          NULL COMMENT '事件',
    `service`     text          NULL COMMENT '服务',
    `creator`     varchar(64)   NULL     DEFAULT '' COMMENT '创建者',
    `create_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updater`     varchar(64)   NULL     DEFAULT '' COMMENT '更新者',
    `update_time` datetime      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `deleted`     bit(1)        NOT NULL DEFAULT b'0' COMMENT '是否删除',
    PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB
  AUTO_INCREMENT = 1
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT = 'IoT 产品物模型功能表';
