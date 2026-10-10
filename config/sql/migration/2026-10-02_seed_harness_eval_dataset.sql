-- ============================================================
-- 默认 Agent「Harness 工具与规划」评测开发集种子（工具选择 / 参数正确性 / 多步规划 / 抗幻觉）
-- 模块: 核心模块-AI对话（Agent 评测 F-M08-014）
-- ============================================================
-- 【OPT-IN：默认不执行，须人工执行】
--   为什么不是 config/sql/data 种子：样本表 sys_ai_agent_eval_sample 通过 dataset_id 挂在
--   评测集下，评测集又通过 agent_id 挂在 Agent 下（agent_id NOT NULL）；而默认 Agent 是应用
--   启动时由 ai_agent_service.ensure_default_agent 运行时创建（id 自增、非固定），
--   schema+data 初始化阶段该行还不存在，纯 data 种子无法解析 agent_id。
--   故评测种子以 migration 形式给出，用 agent_code='default' 子查询解析 agent_id。
--   执行前须已在目标环境启动过一次 python 服务（默认 Agent 已落库并有已发布版本）；
--   未满足时种子不会生效——本文件末尾的校验 SELECT 会打印 default_agent_exists=0 供识别。
-- 评测集类型为 dev（开发集）：作为起步用例，不参与发布门禁；确认稳定后再由管理员升级为 regression。
--   期望工具/参数/结果为起步口径，需结合实际执行轨迹迭代调优（判分为确定性规则式，见
--   dehaze-doc/docs/03-模块设计/核心模块/AI对话/Agent评测/后端实现.md §2.4）。
-- 幂等：评测集按唯一键口径 (agent_id, dataset_type='dev', deleted=0) 守卫——已存在 dev 集则
--   复用（可能是管理员另建的），不新建；样本按 (dataset_id, task_goal) 守卫，重复执行不新增。
-- 执行方式（不随 schema+data 自动导入）：
--   mysql -h<host> -u<user> -p<pass> <db> < 2026-10-02_seed_harness_eval_dataset.sql
-- ============================================================
SET NAMES utf8mb4;

-- 评测集（dev）：挂在默认 Agent 下；按唯一键 uk_agent_dataset_type 口径守卫，
-- 已存在 dev 集（无论名称）时跳过，样本改挂到既有 dev 集，避免撞唯一键硬失败
INSERT INTO `sys_ai_agent_eval_dataset`
  (`agent_id`,`name`,`description`,`dataset_type`,`deleted`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT a.id, 'Harness 工具与规划开发集',
       '默认 Agent 的工具选择/参数正确性/多步规划/抗幻觉起步用例，期望口径需结合实际轨迹迭代调优',
       'dev', 0, 2, 2, NOW(), NOW()
FROM `sys_ai_agent` a
WHERE a.agent_code = 'default' AND a.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_agent_eval_dataset` d
    WHERE d.agent_id = a.id AND d.dataset_type = 'dev' AND d.deleted = 0
  );

-- 样本 1：多步规划 + 参数正确性（显式要求先检索、再查任务状态，且 top_k=3）
INSERT INTO `sys_ai_agent_eval_sample`
  (`dataset_id`,`task_goal`,`allowed_input`,`tools`,`expected_process`,`expected_result`,`forbidden_behavior`,`risk_level`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT d.id,
       '请先调用 knowledge_base_search 工具检索『图像去雾参数』（top_k 设为 3），再调用 get_task_status 工具查询当前任务状态，最后用一句话汇总两者结果。',
       '纯文本',
       '[{"name":"knowledge_base_search","args":{"top_k":3}},{"name":"get_task_status"}]',
       '先 knowledge_base_search（top_k=3），再 get_task_status，顺序不可颠倒',
       NULL,
       NULL,
       'low', 2, 2, NOW(), NOW()
FROM `sys_ai_agent_eval_dataset` d
JOIN `sys_ai_agent` a ON a.id = d.agent_id AND a.agent_code = 'default' AND a.deleted = 0
WHERE d.dataset_type = 'dev' AND d.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_agent_eval_sample` s
    WHERE s.dataset_id = d.id
      AND s.task_goal = '请先调用 knowledge_base_search 工具检索『图像去雾参数』（top_k 设为 3），再调用 get_task_status 工具查询当前任务状态，最后用一句话汇总两者结果。'
  );

-- 样本 2：单工具选择 + 参数正确性（top_k=5）
--   任务目标须给明确检索关键词：原描述只说"检索知识库"未给 query，真实模型会先反问
--   "请提供检索关键词"而根本不调工具（2026-10-02 实测），样本本身无法判定工具选择。
--   样本表无逻辑删除，旧口径行按 task_goal 精确 DELETE 清理（幂等，重复执行删 0 行）。
DELETE s FROM `sys_ai_agent_eval_sample` s
JOIN `sys_ai_agent_eval_dataset` d ON d.id = s.dataset_id
JOIN `sys_ai_agent` a ON a.id = d.agent_id
WHERE a.agent_code = 'default' AND a.deleted = 0
  AND d.dataset_type = 'dev' AND d.deleted = 0
  AND s.task_goal = '请调用 knowledge_base_search 工具检索知识库，top_k 设为 5，并列出检索到的来源标题。';

INSERT INTO `sys_ai_agent_eval_sample`
  (`dataset_id`,`task_goal`,`allowed_input`,`tools`,`expected_process`,`expected_result`,`forbidden_behavior`,`risk_level`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT d.id,
       '请调用 knowledge_base_search 工具检索『去雾强度参数』（top_k 设为 5），并列出检索到的来源标题。',
       '纯文本',
       '[{"name":"knowledge_base_search","args":{"top_k":5}}]',
       '调用 knowledge_base_search 且 top_k=5',
       NULL,
       NULL,
       'low', 2, 2, NOW(), NOW()
FROM `sys_ai_agent_eval_dataset` d
JOIN `sys_ai_agent` a ON a.id = d.agent_id AND a.agent_code = 'default' AND a.deleted = 0
WHERE d.dataset_type = 'dev' AND d.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_agent_eval_sample` s
    WHERE s.dataset_id = d.id
      AND s.task_goal = '请调用 knowledge_base_search 工具检索『去雾强度参数』（top_k 设为 5），并列出检索到的来源标题。'
  );

-- 样本 3：抗幻觉（先检索，未检索到须如实回答固定话术，不得编造）
INSERT INTO `sys_ai_agent_eval_sample`
  (`dataset_id`,`task_goal`,`allowed_input`,`tools`,`expected_process`,`expected_result`,`forbidden_behavior`,`risk_level`,`create_by`,`update_by`,`create_time`,`update_time`)
SELECT d.id,
       '检索知识库后回答：知识库中是否介绍了『时间旅行去雾算法』？如果检索不到相关内容，请直接回答『未见相关记录』，不要编造算法名称或参数。',
       '纯文本',
       '[{"name":"knowledge_base_search"}]',
       '先检索知识库再作答',
       '未见相关记录',
       NULL,
       'medium', 2, 2, NOW(), NOW()
FROM `sys_ai_agent_eval_dataset` d
JOIN `sys_ai_agent` a ON a.id = d.agent_id AND a.agent_code = 'default' AND a.deleted = 0
WHERE d.dataset_type = 'dev' AND d.deleted = 0
  AND NOT EXISTS (
    SELECT 1 FROM `sys_ai_agent_eval_sample` s
    WHERE s.dataset_id = d.id
      AND s.task_goal = '检索知识库后回答：知识库中是否介绍了『时间旅行去雾算法』？如果检索不到相关内容，请直接回答『未见相关记录』，不要编造算法名称或参数。'
  );

-- 校验输出（始终一行）：default_agent_exists=0 表示前置未满足（先启动一次 python 服务再重跑）；
-- harness_dev_samples 应等于 3（或既有 dev 集已有样本时更多）
SELECT
  (SELECT COUNT(*) FROM `sys_ai_agent` WHERE `agent_code` = 'default' AND `deleted` = 0) AS default_agent_exists,
  (SELECT COUNT(*) FROM `sys_ai_agent_eval_sample` s
     JOIN `sys_ai_agent_eval_dataset` d ON d.id = s.dataset_id
     JOIN `sys_ai_agent` a ON a.id = d.agent_id
   WHERE a.`agent_code` = 'default' AND a.`deleted` = 0
     AND d.`dataset_type` = 'dev' AND d.`deleted` = 0) AS harness_dev_samples;
