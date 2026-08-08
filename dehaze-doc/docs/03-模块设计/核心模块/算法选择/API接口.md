# 算法选择模块 API 接口

## 1. 文档概述

本文档定义 **算法选择** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/algorithms/select`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

> **重要**：除 F-M03-007（算法推荐匹配，供 AI 对话 MCP 调用，需明确契约）外，其余接口详细参数/响应结构可通过 API 文档 MCP 查询，本文档定义接口清单、权限标识与 F-M03-007 契约。
>
> **说明**：收藏和图像特征分析推荐功能分别引用 [收藏管理 API](../基础模块/收藏管理/API接口.md) 和 [推荐管理 API](../基础模块/推荐管理/API接口.md)，不在本模块重复定义。F-M03-007 为本模块独有的关键词/样例推荐匹配入口。

## 2. 接口清单

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/algorithms/select/tree` | GET | 获取算法选择树（仅返回已发布状态的算法） | - | F-M03-001 |
| `/api/v1/algorithms/select/{id}` | GET | 获取算法详情（含样例效果图、评分、使用次数） | - | F-M03-004 |
| `/api/v1/algorithms/select/{id}/test` | POST | 上传自定义图片测试算法效果 | - | F-M03-004 |
| `/api/v1/algorithms/select/search` | GET | 搜索算法（关键词/拼音/标签） | - | F-M03-003 |
| `/api/v1/algorithms/select/compare` | POST | 算法对比（最多 3 个算法） | - | F-M03-006 |
| `/api/v1/algorithms/select/recommend` | POST | 算法推荐匹配（基于关键词/任务类型/样例算法，供 AI 对话 MCP 调用） | - | F-M03-007 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 算法选择无特殊权限标识，登录用户即可访问已发布算法 |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0401` | 请求资源不存在 | 查询的算法不存在或未发布；F-M03-007 的 `sampleAlgorithmId` 不存在 |
| `A0701` | 文件格式不支持 | 测试算法效果时上传的图片格式不支持 |
| `A0702` | 文件大小超限 | 测试算法效果时上传图片超过大小限制 |
| `A0500` | 业务异常 | 算法对比数量超过上限（3 个）；F-M03-007 的 `topN` 超出 1-10 范围 |
| `B0100` | 系统执行超时 | 自定义图片测试算法效果超时 |
| `A0230` | token无效或已过期 | 未登录访问 |

## 5. F-M03-007 算法推荐匹配接口契约

> 本节为 F-M03-007 补充详细请求/响应契约，供 AI 对话模块通过 MCP 调用与本模块前端"智能推荐"场景使用。功能定位与降级策略见 [需求规格 §2.7](./需求规格.md)。

### 5.1 接口定义

`POST /api/v1/algorithms/select/recommend`

基于关键词/任务类型/样例算法匹配候选算法，返回 Top N 推荐列表。

### 5.2 请求体

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `keyword` | string | 否 | 关键词（匹配算法名称/拼音/类型/描述） |
| `taskType` | string | 否 | 任务类型枚举（`dehaze`/`derain`/`desnow`/`lowlight`/`super_resolution`/`denoise`/`inpaint`） |
| `sampleAlgorithmId` | integer | 否 | 样例算法 ID，用于查找相似算法 |
| `topN` | integer | 否 | 返回数量，默认 3，范围 1-10 |

> `keyword` 与 `sampleAlgorithmId` 至少传一个；同时为空时按 `taskType` 评分 Top N 兜底返回。

请求示例：

```json
{
  "keyword": "夜景去雾",
  "taskType": "dehaze",
  "topN": 3
}
```

### 5.3 响应体

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | integer | 匹配结果总数 |
| `items[].algorithmId` | integer | 算法 ID |
| `items[].algorithmName` | string | 算法名称 |
| `items[].taskType` | string | 任务类型 |
| `items[].matchScore` | integer | 匹配度 0-100 |
| `items[].reason` | string | 推荐理由 |
| `items[].algorithmScore` | number | 算法评分（来自元数据/评价聚合，待实现） |
| `items[].predictedCostMs` | integer | 预计耗时 ms（来自预测历史均值，待实现） |
| `items[].usageCount` | integer | 使用次数 |

响应示例：

```json
{
  "code": "00000",
  "msg": "一切ok",
  "data": {
    "total": 2,
    "items": [
      {
        "algorithmId": 101,
        "algorithmName": "FFA-Net",
        "taskType": "dehaze",
        "matchScore": 85,
        "reason": "针对重度雾霾场景表现优异",
        "algorithmScore": 4.2,
        "predictedCostMs": 3200,
        "usageCount": 1280
      }
    ]
  }
}
```

### 5.4 降级与边界

| 场景 | 处理 |
|------|------|
| `keyword` 与 `sampleAlgorithmId` 均为空 | 按 `taskType` 评分 Top N 兜底返回（`taskType` 也为空时返回空列表） |
| `taskType` 为空 | 跨所有任务类型匹配 |
| 匹配结果为空 | `data.total=0`、`data.items=[]`，HTTP 200（非业务错误，由调用方展示空态） |
| 接口超时 | 调用方降级为关键词搜索 `GET /api/v1/algorithms/select/search` |
