-- ============================================================
-- 表名: sys_voice_model
-- 模块: 基础模块-语音交互
-- ============================================================
-- 设计思路:
-- ASR 模型 / TTS 音色注册表，管理语音引擎可用的模型/音色清单。
-- model_id 为模型/音色业务编码（ASR：sensevoice/paraformer；TTS：huayan），是业务引用键。
-- engine_type 区分能力（asr/tts），model_type 区分子类型（ASR：stream 流式/offline 离线；TTS：voice 音色）。
-- provider_id 关联 sys_voice_provider.id，标识模型所属引擎；
--   同一 model_id 可通过不同 provider_id 配置多行，实现"同一模型/音色切换引擎"。
--   唯一约束为 (model_id, provider_id) 联合唯一，而非 model_id 单列唯一。
-- params 为模型参数（JSON）：
--   本地引擎（local）：funasr model_id(如 iic/SenseVoiceSmall)/下载 URL/推理参数；Piper onnx 路径/下载 URL/语速映射/编码格式。
--   云端引擎：厂商模型/音色 ID、采样率、编码等透传参数。
-- 本地引擎模型同样由此表注册（provider_id 指向 local 引擎），funasr_engine/piper_tts_engine
--   由写死常量改为从注册表解析，支持注册/切换多个本地 ASR 模型与 TTS 音色。
-- 配置类表，使用逻辑删除；model_id 为业务引用键，删除后不可复用（类别②，查重查全表）。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `sys_voice_model`;
CREATE TABLE `sys_voice_model`
(
    `id`            bigint                                                          NOT NULL AUTO_INCREMENT COMMENT '主键',
    `provider_id`   bigint                                                          NOT NULL COMMENT '关联引擎ID(关联sys_voice_provider.id)',
    `model_id`      varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '模型/音色业务编码(sensevoice;paraformer;huayan;删除后不可复用)',
    `engine_type`   varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '能力类型(asr:语音识别;tts:语音合成)',
    `model_type`    varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci   NOT NULL COMMENT '子类型(ASR:stream流式/offline离线;TTS:voice音色)',
    `display_name`  varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci  NOT NULL COMMENT '显示名称(如中文女声;SenseVoice流式)',
    `params`        json                                                            NULL COMMENT '模型参数(JSON:本地模型路径/下载URL/推理参数;云端厂商模型ID/采样率/编码等)',
    `status`        tinyint                                                         NOT NULL DEFAULT 1 COMMENT '状态(1:启用;0:禁用)',
    `deleted`       tinyint                                                         NOT NULL DEFAULT 0 COMMENT '逻辑删除标识(0:未删除;1:已删除)',
    `create_by`     bigint                                                          NULL DEFAULT NULL COMMENT '创建人ID',
    `update_by`     bigint                                                          NULL DEFAULT NULL COMMENT '修改人ID',
    `create_time`   datetime                                                        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`   datetime                                                        NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `uk_model_provider` (`model_id`, `provider_id`) USING BTREE,
    INDEX `idx_engine_type` (`engine_type`, `model_type`, `status`) USING BTREE
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci COMMENT = '语音引擎模型/音色注册表'
  ROW_FORMAT = DYNAMIC;
