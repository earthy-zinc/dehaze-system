-- ============================================================
-- 种子数据: sys_ai_provider（AI 模型供应商配置表）
-- 模块: 基础模块-AI模型管理
-- ============================================================
-- 正式供应商清单（固定 id，与 sys_ai_model.sql 的 provider_id 引用一一对应）：
--   id=1   openai      OpenAI         https://api.openai.com/v1
--   id=77  local       内置本地模型    http://127.0.0.1:8992/v1（内置本地 LLM 子进程服务）
--   id=78  local-voice 内置语音引擎    http://127.0.0.1:8993/v1（dehaze-voice，openai_audio）
--   id=79  local-modal 内置图像引擎    http://127.0.0.1:8994/v1（dehaze-modal，openai_image）
--   id=177 deepseek    DeepSeek       https://api.deepseek.com
--   id=180 volc-ark    火山方舟       https://ark.cn-beijing.volces.com/api/v3（openai_compat）
--   id=181 volc-voice  火山豆包语音   https://openspeech.bytedance.com（volc_realtime，asr/tts 共用）
--   id=182 siliconflow 硅基流动       https://api.siliconflow.cn/v1（openai_compat）
-- 说明：
--   1. 本文件为供应商数据的单一信息源，由 rebuild_mysql.py --import 全量同步
--      （先清表再导入），历史测试残留（SDK 集成测试创建的 test_prov_* 供应商）
--      不在清单内，全量同步后即被清除。
--   2. id 显式固定：sys_ai_model.sql 的 provider_id（1/77/78/79/177）与 sys_ai_provider_key
--      的 provider_id 均引用此五值，固定 id 保证引用不失效。
--   3. API Key 不在此播种：key_hash/key_cipher 涉及密钥，由 `python -m app.cli init-builtin`
--      显式初始化（需 AES 加密，无法静态落 SQL），不进入版本库。
--   4. user_identity_forward 全部为 NULL（不启用），需透传时由管理页面按供应商能力配置。
--   5. request_timeout_seconds：请求超时(秒)，NULL=调用方默认；local（CPU 推理慢）显式 300
--   6. 内置供应商（local/local-voice/local-modal）health_check_enabled=0：不参与熔断判定
--      （错误多为瞬时/服务未启动，交由请求方按超时处理）；需要熔断保护时在管理端开启。
--   7. **仅供全新库**：本文件是 plain INSERT + 固定主键，用于全新库（DROP+CREATE 后全量导入）；
--      存量库重跑会 `ERROR 1062 Duplicate entry` 中止（后续行进不去，但不会污染既有行）。
--      存量环境（dev/pre）**不要**重跑本文件、也不要用 `rebuild_mysql.py --import sys_ai_provider.sql`
--      （该入口会先整表 DELETE，清掉管理端自建供应商）；改用幂等 migration
--      `config/sql/migration/2026-10-01_seed_volcano_providers_models.sql`。
-- ============================================================
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('1','openai','OpenAI','https://api.openai.com/v1','openai_compat','bearer',NULL,'0','1',NULL,NULL,'1','0','2','2',NULL,'2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('77','local','内置本地模型','http://127.0.0.1:8992/v1','openai_compat','bearer',NULL,'0','0',NULL,'内置本地 LLM 服务（127.0.0.1:8992），Key 为占位值','1','0','2','2','300','2026-08-21 00:00:00','2026-08-21 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('78','local-voice','内置语音引擎','http://127.0.0.1:8993/v1','openai_audio','bearer',NULL,'0','0',NULL,'内置本地语音服务（127.0.0.1:8993），Key 为服务密钥','1','0','2','2',NULL,'2026-09-29 00:00:00','2026-09-29 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('79','local-modal','内置图像引擎','http://127.0.0.1:8994/v1','openai_image','bearer',NULL,'0','0',NULL,'内置本地图像服务（127.0.0.1:8994），Key 为服务密钥','1','0','2','2',NULL,'2026-09-29 00:00:00','2026-09-29 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('177','deepseek','DeepSeek（外部真实模型）','https://api.deepseek.com','openai_compat','bearer',NULL,'10','1',NULL,'真实外部模型测试接入','1','0','2','2',NULL,'2026-08-21 00:00:00','2026-08-21 00:00:00');
-- 第三方模型供应商（火山引擎方案）：
--   volc-ark   chat/embedding/rerank（方舟 OpenAI 兼容 v3，chat 超时 120s）
--   volc-voice asr/tts 共用（volc_realtime 二进制帧协议；api_base_url 只取 scheme+host，适配器忽略路径拼固定端点）
--              default_headers 键义：
--                auth_header      新/旧控制台口径开关——置头名（如 X-Api-Key）走单密钥口径，留空走经典口径
--                X-Api-App-Key    经典口径 ASR 的 APP ID（非密钥但需按账号填写，此处留空由管理端补齐）
--                asr_resource_id  ASR 资源 ID（协议固定值 volc.seedasr.sauc.duration）
--                tts_resource_id  TTS 资源 ID（单密钥 v3 端点用，协议固定值 seed-tts-2.0）
--                cluster          经典口径 TTS 集群名（缺省 volcano_tts）
--               资源 ID 按能力分列：同一供应商行服务 asr/tts 两条链路，二者取值不同，故各占一键
--   siliconflow embedding/rerank（硅基流动 OpenAI 兼容 v1，超时 30s）
--   云厂商 health_check_enabled=1（参与熔断），区别于内置 local 系供应商（=0）
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('180','volc-ark','火山方舟','https://ark.cn-beijing.volces.com/api/v3','openai_compat','bearer',NULL,'20','1',NULL,'火山引擎方舟大模型（chat/embedding/rerank）','1','0','2','2','120','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('181','volc-voice','火山豆包语音','https://openspeech.bytedance.com','volc_realtime','bearer','{"auth_header":"X-Api-Key","X-Api-App-Key":"","asr_resource_id":"volc.seedasr.sauc.duration","tts_resource_id":"seed-tts-2.0","cluster":"volcano_tts"}','21','1',NULL,'火山引擎流式语音识别/合成（asr/tts 共用供应商；单密钥 X-Api-Key 由管理端加密写入密钥表）','1','0','2','2','60','2026-10-01 00:00:00','2026-10-01 00:00:00');
INSERT INTO `sys_ai_provider` (`id`,`provider_code`,`display_name`,`api_base_url`,`protocol_type`,`auth_type`,`default_headers`,`sort_order`,`health_check_enabled`,`user_identity_forward`,`remark`,`status`,`deleted`,`create_by`,`update_by`,`request_timeout_seconds`,`create_time`,`update_time`)  VALUES ('182','siliconflow','硅基流动','https://api.siliconflow.cn/v1','openai_compat','bearer',NULL,'22','1',NULL,'硅基流动（embedding/rerank，免费额度）','1','0','2','2','30','2026-10-01 00:00:00','2026-10-01 00:00:00');

SET FOREIGN_KEY_CHECKS = 1;
