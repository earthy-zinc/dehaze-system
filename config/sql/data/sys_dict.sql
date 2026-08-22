SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_dict (id, type_code, name, value, sort, status, defaulted, remark, create_time, update_time)
values (1, 'gender', '男', '1', 1, 1, 0, null, '2019-05-05 13:07:52', '2022-06-12 23:20:39'),
       (2, 'gender', '女', '2', 2, 1, 0, null, '2019-04-19 11:33:00', '2019-07-02 14:23:05'),
       (3, 'gender', '未知', '0', 1, 1, 0, null, '2020-10-17 08:09:31', '2020-10-17 08:09:31'),
       -- AI 推理参数系统默认（后端实现 §10.2；Agent 配置为空的键继承本默认，不硬编码在代码中）
       (4, 'ai_reasoning_defaults', 'max_steps_react', '20', 1, 1, 1, 'ReAct 最大推理步数', NOW(), NOW()),
       (5, 'ai_reasoning_defaults', 'max_steps_plan', '30', 2, 1, 1, 'Plan-and-Execute 最大推理步数', NOW(), NOW()),
       (6, 'ai_reasoning_defaults', 'max_steps_reflexion', '15', 3, 1, 1, 'Reflexion 单次迭代最大步数', NOW(), NOW()),
       (7, 'ai_reasoning_defaults', 'max_iterations_reflexion', '3', 4, 1, 1, 'Reflexion 最大迭代次数', NOW(), NOW()),
       (8, 'ai_reasoning_defaults', 'reflexion_threshold', '0.8', 5, 1, 1, 'Reflexion 质量达标阈值', NOW(), NOW()),
       (9, 'ai_reasoning_defaults', 'max_parallel', '5', 6, 1, 1, '并行子任务最大数', NOW(), NOW()),
       (10, 'ai_reasoning_defaults', 'tool_timeout', '60', 7, 1, 1, '单工具调用超时（秒）', NOW(), NOW()),
       (11, 'ai_reasoning_defaults', 'token_budget', '500000', 8, 1, 1, '单会话 Token 预算上限', NOW(), NOW()),
       (12, 'ai_reasoning_defaults', 'retry_max', '2', 9, 1, 1, '工具调用失败最大重试次数', NOW(), NOW()),
       -- AI 护栏系统默认（后端实现 §10.3；点分键在 resolver 加载时组装为嵌套 {规则: {参数}}）
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
       (26, 'ai_embedding', 'dims', '1536', 3, 1, 1, '向量维度(ES dense_vector dims 联动)', NOW(), NOW());

SET FOREIGN_KEY_CHECKS = 1;