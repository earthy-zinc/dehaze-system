# 会员管理模块 - API接口

## 1. 文档概述

本文档定义 **会员管理** 模块的 HTTP API 规范，是该模块 API 契约的**唯一权威来源**。

- **基础路径**：`/api/v1/members`
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)
- **需求规格**：[需求规格.md](./需求规格.md)

## 2. 接口清单

### 2.1 用户端接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/members/profile` | GET | 当前用户会员基础信息（等级、等级来源、成长值、到期时间、状态、进度） | - | F-MM-012 |
| `/api/v1/members/benefit-summary` | GET | 权益概览：按服务类目合并展示（图像处理 / 评估剩余次数、AI 积分余额与额度），含类目明细 | - | F-MM-012 |
| `/api/v1/members/trial-status` | GET | 试用引导状态：体验券激活状态与到期时间、AI 试用积分余额、新用户专享可用性 | - | F-MM-015 |
| `/api/v1/members/growth-logs` | GET | 成长值变动明细 | - | F-MM-013 |
| `/api/v1/members/sign-in` | POST | 每日签到 | - | F-MM-014 |
| `/api/v1/members/sign-in/calendar` | GET | 签到日历 | - | F-MM-014 |

### 2.2 后台管理接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/members/page` | GET | 会员分页列表 | member:list | F-MM-006 |
| `/api/v1/members/{userId}` | GET | 会员详情 | - | F-MM-007 |
| `/api/v1/members/{userId}/level` | PUT | 等级调整 | member:level:edit | F-MM-008 |
| `/api/v1/members/{userId}/growth` | PUT | 成长值调整 | member:growth:edit | F-MM-009 |
| `/api/v1/members/{userId}/status` | PUT | 冻结/解冻 | member:status:edit | F-MM-010 |
| `/api/v1/members/benefits` | GET | 权益配置列表 | - | F-MM-011 |
| `/api/v1/members/benefits/{level}` | PUT | 修改权益配置 | member:benefit:edit | F-MM-011 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| member:list | 会员分页列表 |
| member:level:edit | 等级调整 |
| member:growth:edit | 成长值调整 |
| member:status:edit | 冻结/解冻 |
| member:benefit:edit | 权益配置 |

> 权益配置（sys_member_benefit）字段含 `ai_credits_daily`（AI 日限额）、`ai_credits_monthly`（AI 月限额）、`multimodal_limit`（多模态视觉读取日限额）、`vip_gift_credits`（VIP 按月赠送积分）等 AI 计费相关配额列，权益配置接口返回全部限额字段。

> 用户端接口（会员信息 / 权益概览 / 试用引导 / 成长值明细 / 签到 / 签到日历）与后台查询接口（会员列表 / 详情 / 权益配置列表）均为登录态访问。

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| A0510 | 会员不存在 | 查询/操作时会员记录不存在 |
| A0511 | 会员已冻结 | 冻结状态会员尝试使用付费功能 |
| A0512 | 今日已签到 | 重复签到 |
| A0513 | 成长值不足 | 扣减成长值时余额不足 |
| A0514 | 权益配置无效 | 权益配置缺少必填项或数值非法 |
