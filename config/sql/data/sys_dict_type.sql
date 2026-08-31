SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO `sys_dict_type` (id, name, code, status, remark, create_time, update_time)
VALUES (1, '性别', 'gender', 1, NULL, '2019-12-06 19:03:32', '2022-06-12 16:21:28'),
       (3, 'AI护栏系统默认', 'ai_guardrail_defaults', 1, '安全护栏开关全局默认值（后端实现 §10.3），Agent 级 config.guardrails 逐项覆盖', NOW(), NOW()),
       (4, 'AI供应商健康阈值', 'ai_provider_health', 1, '供应商健康与熔断阈值全局默认（后端实现 §2.4），错误率/连续失败/熔断冷却等', NOW(), NOW()),
       (5, 'AI记忆Embedding配置', 'ai_embedding', 1, '记忆向量化 Embedding 全局配置（后端实现 §7.7），provider_code/model/dims 三键', NOW(), NOW()),
       (6, '会员成长值规则', 'member_growth_rules', 1, '营销激励参数，运营可调', NOW(), NOW()),
       (7, '收藏容量', 'favorite_capacity', 1, '各会员等级收藏容量上限，运营可调', NOW(), NOW()),
       (8, 'AI评测参数', 'ai_eval', 1, '智能体评测质量参数（评测中心/判分治理），退化阈值/一致性阈值/复核抽样比例', NOW(), NOW());

SET FOREIGN_KEY_CHECKS = 1;