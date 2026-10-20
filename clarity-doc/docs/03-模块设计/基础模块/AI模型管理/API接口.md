# AI 模型管理模块 API 接口

## 文档概述

模块承载平台全部模态的 AI 模型（chat/embedding/rerank/image_gen/image_edit/video_gen/asr/tts 及预留 music_gen/3d_gen）与模型供应商（含 API Key）的统一管理。

- **基础路径**：`/api/v1/ai`
- **消费方**：AI 对话（会话内 chat 模型选择）、AI 知识库（按 `model_type` 选择 embedding/rerank 模型）、去雾处理（`modelType=image_edit` 图像模型选择）、语音交互（`model_type=asr/tts` 语音模型路由）

## 接口清单

### 模型供应商管理接口（AiProviderAPI）

供应商与 API Key 接口均需 `ai:model:manage` 权限，`/providers/enabled` 除外。供应商不限于对话 LLM，覆盖全部模型类型（含图像、语音）的供应商接入；同一厂商一行即可提供其全部类型模型。

| 路径                                        | 方法   | 功能描述                                                 | 权限标识          | 关联功能点 |
| ------------------------------------------- | ------ | -------------------------------------------------------- | ----------------- | ---------- |
| `/api/v1/ai/providers`                      | GET    | 供应商分页列表（含健康状态：健康、可疑、熔断）             | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/enabled`              | GET    | 启用供应商列表（普通用户只读，不含敏感信息）             | -                 | F-M08-007  |
| `/api/v1/ai/providers`                      | POST   | 新增供应商（保存后异步触发连通性测试，结果仅提示不阻断） | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}`                 | PUT    | 更新供应商                                               | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}`                 | DELETE | 删除供应商（逻辑删除）                                   | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/test-connection` | POST   | 连通性测试（不产生计费）                                 | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/circuit/close`   | POST   | 手动解除供应商熔断状态                                   | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/keys`            | GET    | 供应商下的 API Key 列表（不返回明文）                    | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/keys`            | POST   | 新增 API Key（请求体携带明文，服务端加密存储）           | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/keys/{keyId}`    | PUT    | 更新 API Key（优先级、权重、状态、限额）                 | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/providers/{id}/keys/{keyId}`    | DELETE | 删除 API Key（物理删除，校验至少保留一个启用 Key）       | `ai:model:manage` | F-M08-007  |

### 模型管理接口（AiModelAPI）

模型管理分双端：管理端分页列表需 `ai:model:manage` 权限；`/models/enabled` 无需特殊权限，按登录用户 VIP 等级过滤。两者均支持 `modelType` 查询参数按模型类型筛选（`chat`/`embedding`/`rerank`/`image_gen`/`image_edit`/`video_gen`/`asr`/`tts` 等）；图像模型选择场景使用 `?modelType=image_edit`（可按 `taskTags` 进一步过滤任务标签）。更新、删除、测试与价格版本以 **`model_id` 字符串**为业务键（非自增主键），因厂商模型名含 `/`（如 `Qwen/Qwen3-8B`）不能作路径段，一律经 query 参数 `modelId` 传递；新增价格版本的 `modelId` 随请求体提交。

| 路径                              | 方法   | 功能描述                                                          | 权限标识          | 关联功能点 |
| --------------------------------- | ------ | ----------------------------------------------------------------- | ----------------- | ---------- |
| `/api/v1/ai/models`               | GET    | 模型分页列表（管理端，支持 `modelType` 筛选）                     | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models/enabled`       | GET    | 启用模型列表（用户端、消费方，按 VIP 过滤，支持 `modelType` 筛选） | -                 | F-M08-007  |
| `/api/v1/ai/models`               | POST   | 新增模型配置                                                      | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models?modelId=`      | PUT    | 更新模型配置（`model_type` / `dimension` 创建后不可修改）         | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models?modelId=`      | DELETE | 删除模型配置（逻辑删除，`model_id` 不可复用）                     | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models/test?modelId=` | POST   | 模型可用性测试（发最小真实推理请求并落库结果）                    | `ai:model:manage` | F-M08-007  |

模型分页列表（GET `/api/v1/ai/models`）每项附带运行时状态字段：`lastTestStatus`（0-未测试/1-可用/2-不可用）、`lastTestAt`、`lastTestError`（最近可用性测试结果），以及近 24h 真实调用统计 `calls24h`（null=未调用）、`successRate24h`（成功率百分比）、`lastCallAt`。

**平台默认模型标记 `isDefault`（仅启用模型列表返回）**：`GET /api/v1/ai/models/enabled` 每项带 `isDefault`（布尔），表示该模型是否为其 `model_type` 的平台默认模型——取该类型对应字典键 `ai_model` 组的 `default_model`（chat）/ `default_model.{model_type}`（如 `default_model.image_edit`）的启用值（`status=0` 视同未配置），与该值 `model_id` 相同的行标记为 `true`，其余为 `false`；同一 `model_id` 配多供应商时该模型各行均为 `true`。该字段为读取时按 `model_type` 分键计算（不写入跨端共享缓存快照 `ai:model:list`，不叠加列表缓存 TTL 带来的滞后），故不改变启用列表既有的排序与过滤行为。消费者（图像、语音、向量选择页）在无历史记录时以此预选后端默认模型，不得用"启用列表首项"等猜值兜底；管理端分页列表（GET `/api/v1/ai/models`）不返回该字段（键缺失 ≡ 未计算，勿按 `false` 解读）。

### 模型用户售价接口（AiModelAPI）

用户售价按版本管理：同模型同供应商新增即生成新版本（版本号递增），历史版本保留可追溯。**token 模型**（chat/embedding/rerank）每个版本含三维档位明细（token 类型 `input`/`cached`/`output` × 时段 `peak`/`idle` × 上下文分段 `min_tokens`~`max_tokens`），主表 `billing_unit=million_tokens`；**非 token 模型**（image/asr/tts）主表 `billing_unit` 为 `image`/`second`/`ten_thousand_chars`，detail 表单行 `unit` 档（`usage_kind='unit'`，不设峰谷）。分页参数为 `page`/`size`，与管理端分页的 `pageNum`/`pageSize` 不同。

| 路径                                  | 方法   | 功能描述                                       | 权限标识          | 关联功能点 |
| ------------------------------------- | ------ | ---------------------------------------------- | ----------------- | ---------- |
| `/api/v1/ai/models/prices?modelId=`   | GET    | 价格版本分页列表（含档位明细）                 | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models/prices`            | POST   | 新增价格版本（生成新版本号，含档位明细）       | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models/prices/{price_id}` | PUT    | 更新价格版本主表字段（单价单位、生效时间、状态） | `ai:model:manage` | F-M08-007  |
| `/api/v1/ai/models/prices/{price_id}` | DELETE | 删除价格版本（主表与档位明细一并逻辑删除）     | `ai:model:manage` | F-M08-007  |

档位单价 `unit_price` 后端为 `Decimal`，响应序列化为字符串，前端参与计算前需转 Number；档位明细不支持局部更新，调价走新增价格版本。

### 运营统计接口

| 路径                     | 方法 | 功能描述                                                                                      | 权限标识          | 关联功能点 |
| ------------------------ | ---- | --------------------------------------------------------------------------------------------- | ----------------- | ---------- |
| `/api/v1/ai/usage/stats` | GET  | 用量与成本统计（供应商健康看板、模型用量分布、降级与故障统计），支持时间范围（start/end）过滤 | `ai:model:manage` | F-M08-007  |

`providerHealth` 各项：`callCount` 为近 24h 调用量；`successRate`（成功率 0-1）、`rate429`（429 限流率 0-1）、`p95LatencyMs`（延迟 P95 毫秒）由该窗口聚合，**窗口内无调用时三者无样本**——python 响应信封剔除 null 字段（键缺失 ≡ null），java/go 输出 `null`，客户端按「无数据」展示（Web 端显示 `-`），不得当作 100% 成功 / 0ms。

### 用户偏好默认模型接口（AiUserPreferenceAPI）

登录用户本人读写，无权限码，PUT 为全量替换语义：`defaultModel` 传 `null` 或省略该字段均视同清除偏好（回落平台默认模型）；非空值须为本人可见的启用 chat 模型，否则返回 `A0601`；空串 `""` 不是清除，按非法模型返回 `A0601`。响应体统一剔除 null：未设置偏好时 `defaultModel` 键不出现（键缺失 ≡ null ≡ 跟随平台默认模型），已设置时为字符串值。偏好只决定新建会话与请求未指定模型时的默认模型，不改变已有会话的模型。

| 路径                         | 方法 | 功能描述                                                                                  | 权限标识 | 关联功能点 |
| ---------------------------- | ---- | ----------------------------------------------------------------------------------------- | -------- | ---------- |
| `/api/v1/ai/user-preference` | GET  | 查询本人偏好默认模型（无记录、未设置时响应不含 `defaultModel` 键，不自动建行）             | -        | F-M08-007  |
| `/api/v1/ai/user-preference` | PUT  | 保存本人偏好默认模型（全量替换：`defaultModel` 传 `null` 或省略该字段即清除，返回空响应） | -        | F-M08-007  |

PUT 的 `defaultModel` 非空（含空串）时按 `GET /api/v1/ai/models/enabled?modelType=chat` 同一可见性规则校验，不满足返回 `A0601`；空串匹配不到任何模型 id，故按非法模型处理。

## 权限标识汇总

| 权限标识          | 说明                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| `ai:model:manage` | 模型与供应商配置管理（模型、供应商、API Key 的新增、修改、删除及运营统计），仅管理员；越权返回 403（`A0301`） |
| -                 | 启用模型、供应商列表查询，登录用户可用；消费方（AI 对话、AI 知识库）按模型类型过滤使用                       |
| -                 | 用户偏好默认模型读写（仅本人），登录用户可用                                                                |

## 业务错误码

| 错误码  | 说明                   | 触发场景                                                                                                                                                                              |
| ------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `A0230` | token 无效或已过期     | 未登录访问                                                                                                                                                                            |
| `A0301` | 访问未授权             | 无 `ai:model:manage` 权限调用管理端接口                                                                                                                                               |
| `A0401` | 请求资源不存在         | 模型、供应商/API Key 不存在                                                                                                                                                            |
| `A0500` | 业务异常               | 未配置平台默认模型（`sys_dict` 组 `ai_model` 的 `default_model`（chat）/`default_model.asr`/`.tts`/`.image_edit` 等按类型键缺项、值为空或字典项被停用），对应用户端调用报错并提示配置 |
| `A0501` | 数据已存在             | `model_id` / `provider_code` / API Key 明文查重命中（含逻辑删除历史，不可复用）                                                                                                       |
| `A0504` | 存在关联数据，无法删除 | 删除被活跃会话引用的模型、被启用模型降级链（`fallback_model_id`）引用的模型、存在启用模型引用的供应商                                                                                 |
| `A0601` | 模型不可用             | 偏好默认模型不存在、未启用或当前等级不可用（模型选择不满足能力要求的场景见 [AI 对话/API接口](../../核心模块/AI对话/API接口.md)）                                                      |
