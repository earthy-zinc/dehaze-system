-- ============================================================
-- 表名: sys_voice_provider
-- 模块: 基础模块-语音交互
-- ============================================================
-- 设计思路:
-- 语音引擎（ASR/TTS）供应商配置表，对齐 AI 模型管理的 sys_ai_provider 模式。
-- 本地（FunASR/Piper）与云端（阿里云/腾讯云/讯飞 ASR、Azure/阿里云 TTS）统一注册，
-- 应用侧透明，不区分本地/云端，按能力维度（engine_type）选择默认引擎调用。
-- (provider_code, engine_type) 为业务唯一键（同一厂商按能力注册多条，如 local 注册 asr、tts 两条），删除后不可复用（类别②）。
-- engine_type 标识能力类型（asr/tts）；asr/tts 共用此表，不拆表（provider 层属性同构）。
-- is_default 标识该 engine_type 维度下默认引擎（每能力维度仅一条为 1），默认指向 local；
--   纯云端部署将 asr/tts 的 is_default 指向云端引擎（local 仍保留可选用）。
-- api_base_url 为引擎 API 端点（local 为空，走进程内 FunASR/Piper 引擎）。
-- auth_type 决定 HTTP 认证头格式：bearer、x-api-key 或 custom（头名在 default_headers 配置）。
-- default_headers 存储引擎特有请求头，JSON 格式。
-- health_check_enabled 为健康检查开关（默认开启），关闭后该引擎不参与熔断判定（健康状态运行时聚合于 Redis，不落库）。
-- remark 为运维备注（账号归属、合同号、商务信息），不参与逻辑。
-- 引擎下的 API Key 管理见 sys_voice_provider_key，模型/音色注册见 sys_voice_model。
-- 配置类表，使用逻辑删除；provider_code 为业务引用键，删除后不可复用（类别②）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_voice_provider`;
CREATE TABLE `sys_voice_provider`
(
    `id`                   bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `provider_code`        varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '引擎编码(local;aliyun;tencent;xfyun;azure;与engine_type联合唯一,同一厂商按能力注册多条;删除后不可复用)',
    `engine_type`          varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '能力类型(asr:语音识别;tts:语音合成)',
    `display_name`         varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '显示名称(如本地FunASR;阿里云ASR;本地Piper;Azure TTS)',
    `api_base_url`         varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL COMMENT '引擎API基础地址(local为空,走进程内引擎)',
    `auth_type`            varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL DEFAULT 'bearer' COMMENT '认证方式(bearer:Authorization Bearer;x-api-key;custom:自定义请求头,头名在default_headers配置)',
    `default_headers`      json                                                            NULL COMMENT '默认请求头(JSON);auth_type=custom时,需含{"auth_header":"头名"}',
    `is_default`           tinyint                                                         NOT NULL DEFAULT 0 COMMENT '该engine_type维度下默认引擎(0:否;1:是;每能力维度仅一条为1)',
    `sort_order`           int                                                             NOT NULL DEFAULT 0 COMMENT '排序序号(数字越小越靠前)',
    `health_check_enabled` tinyint                                                         NOT NULL DEFAULT 1 COMMENT '健康检查开关(1:开启,参与熔断判定;0:关闭)',
    `remark`               varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NULL DEFAULT NULL COMMENT '运维备注(账号归属/合同号/商务信息)',
    `status`               tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`              tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`            bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`            bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`          datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`          datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_provider_engine` (`provider_code`, `engine_type`) USING BTREE,
    INDEX `idx_engine_default` (`engine_type`, `is_default`) USING BTREE,
    INDEX `idx_engine_status` (`engine_type`, `status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '语音引擎供应商配置表'
  ROW_FORMAT = DYNAMIC;
