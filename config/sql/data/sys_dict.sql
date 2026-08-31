SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_dict (id, type_code, name, value, sort, status, defaulted, remark, create_time, update_time)
values (1, 'gender', '男', '1', 1, 1, 0, null, '2019-05-05 13:07:52', '2022-06-12 23:20:39'),
       (2, 'gender', '女', '2', 2, 1, 0, null, '2019-04-19 11:33:00', '2019-07-02 14:23:05'),
       (3, 'gender', '未知', '0', 1, 1, 0, null, '2020-10-17 08:09:31', '2020-10-17 08:09:31'),
       -- AI 护栏系统默认（后端实现 §10.3；点分键在 resolver 加载时组装为嵌套 {规则: {参数}}）
       -- 注：推理参数默认值为代码常量（agent_config_resolver.REASONING_DEFAULTS），不入 sys_dict
       (13, 'ai_guardrail_defaults', 'prompt_injection.enabled', 'true', 1, 1, 1, 'Prompt 注入防护开关', NOW(), NOW()),
       (14, 'ai_guardrail_defaults', 'unauthorized_access.enabled', 'true', 2, 1, 1, '越权查询检测开关', NOW(), NOW()),
       (15, 'ai_guardrail_defaults', 'sensitive_topic.enabled', 'false', 3, 1, 1, '敏感话题过滤开关', NOW(), NOW()),
       (16, 'ai_guardrail_defaults', 'pii_mask.enabled', 'true', 4, 1, 1, '敏感信息脱敏开关', NOW(), NOW()),
       (17, 'ai_guardrail_defaults', 'fact_check.enabled', 'false', 5, 1, 1, '事实性校验开关', NOW(), NOW()),
       (18, 'ai_guardrail_defaults', 'format_check.enabled', 'false', 6, 1, 1, '格式合规校验开关', NOW(), NOW()),
       -- AI 供应商健康与熔断阈值（后端实现 §2.4；错误率阈值10%/30%、连续失败≥5、熔断冷却60s）
       (19, 'ai_provider_health', 'error_rate_warn', '0.1', 1, 1, 1, '可疑错误率阈值(≥即为可疑)', NOW(), NOW()),
       (20, 'ai_provider_health', 'error_rate_open', '0.3', 2, 1, 1, '熔断错误率阈值(≥即为熔断)', NOW(), NOW()),
       (21, 'ai_provider_health', 'min_window_calls', '20', 3, 1, 1, '错误率判定最小调用窗口', NOW(), NOW()),
       (22, 'ai_provider_health', 'consecutive_failures', '5', 4, 1, 1, '连续失败熔断阈值', NOW(), NOW()),
       (23, 'ai_provider_health', 'circuit_cooldown', '60', 5, 1, 1, '熔断冷却时长(秒)', NOW(), NOW()),
       -- 记忆向量化 Embedding 配置（后端实现 §7.7；provider_code/model/dims 三键，dims 与 ES mapping 联动）
       (24, 'ai_embedding', 'provider_code', 'openai', 1, 1, 1, 'Embedding 供应商编码(经 ai_provider 体系取 Key)', NOW(), NOW()),
       (25, 'ai_embedding', 'model', 'text-embedding-3-small', 2, 1, 1, 'Embedding 模型标识', NOW(), NOW()),
       (26, 'ai_embedding', 'dims', '1536', 3, 1, 1, '向量维度(ES dense_vector dims 联动)', NOW(), NOW()),
       -- 会员成长值规则（后端实现 §9.1；营销激励参数，运营可调）
       (27, 'member_growth_rules', 'sign_in_value', '3', 1, 1, 1, '每日签到获得成长值', NOW(), NOW()),
       (28, 'member_growth_rules', 'sign_in_streak_bonus', '20', 2, 1, 1, '连续签到奖励（连续7天额外获得）', NOW(), NOW()),
       (29, 'member_growth_rules', 'rating_growth_value', '5', 3, 1, 1, '单次评价获得成长值', NOW(), NOW()),
       (30, 'member_growth_rules', 'rating_growth_daily_limit', '5', 4, 1, 1, '评价成长值上限（每日评价获得成长值次数上限）', NOW(), NOW()),
       -- 收藏容量（后端实现 §11.1；各会员等级收藏容量上限）
       (31, 'favorite_capacity', 'default', '200', 1, 1, 1, '普通用户(level_0)收藏容量', NOW(), NOW()),
       (32, 'favorite_capacity', 'vip1', '500', 2, 1, 1, 'VIP1(level_1)收藏容量', NOW(), NOW()),
       (33, 'favorite_capacity', 'vip2', '1000', 3, 1, 1, 'VIP2(level_2)收藏容量', NOW(), NOW()),
       (34, 'favorite_capacity', 'svip', '3000', 4, 1, 1, 'SVIP(level_3)收藏容量', NOW(), NOW()),
       -- AI 评测质量参数（评测中心 F-M08-014；阈值均为百分比/百分制整数，运营可调）
       (35, 'ai_eval', 'regression_threshold', '5', 1, 1, 1, '相对退化阈值(%,相对上次评测总分下降超此值判定退化)', NOW(), NOW()),
       (36, 'ai_eval', 'judge_consistency_threshold', '90', 2, 1, 1, '判分一致性阈值(%,人工复核一致率低于此值判定漂移)', NOW(), NOW()),
       (37, 'ai_eval', 'judge_review_ratio', '1', 3, 1, 1, '人工复核抽样比例(%,通过样本按此比例确定性抽样)', NOW(), NOW());

SET FOREIGN_KEY_CHECKS = 1;