// Package aidomain 实现 AI 对话域 A 类端点的原生业务逻辑（对齐 dehaze-python 行为）。
// B 类（推理/流式）由 python 承接，Go 侧只做转发（见 internal/router/ai_proxy.go）。
package aidomain

import (
	"context"
	"encoding/json"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
)

const timeFormat = "2006-01-02 15:04:05"

// rawJSON 将库中的 JSON 字符串转为可直接序列化的原始消息（空串→null）。
func rawJSON(s string) json.RawMessage {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}

// marshalJSON 将结构序列化为库中的 JSON 字符串。
func marshalJSON(v any) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

// jsonColumnValue 生成"可安全写入 JSON 列"的值：nil、或序列化后为空/null 的输入 → nil（落 SQL NULL），
// 其余 → JSON 文本。专供 map 型更新（`Updates(map)` 不跳过零值，若写空串会触发 MySQL 3140
// "Invalid JSON text: The document is empty"）；对应 python 侧 `snapshot.get("permissions")`
// 的 None→NULL 语义。结构体字段赋值仍用 marshalJSON（零值由 model 的 `default:null` 兜底省略列）。
func jsonColumnValue(v any) any {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil || len(b) == 0 || string(b) == "null" {
		return nil
	}
	return string(b)
}

func formatTime(t time.Time) string { return t.Format(timeFormat) }

func formatTimePtr(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.Format(timeFormat)
}

func derefString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

// ── 跨端共享缓存键（与 dehaze-python app/service/ai_agent_service.py 完全一致）──
//
// 三端共享同一批 Redis 键：python 读多级缓存（L1 进程内 + L2 Redis），Go 写操作必须
// 在事务提交后删 L2 并向 cache:invalidation 频道广播，否则 python 实例的 L1 脏值会
// 存活到 TTL 到期。
const (
	agentDetailKeyFmt    = "ai:agent:%s"
	agentSkillKeyFmt     = "ai:agent:%d:skills"
	agentMcpKeyFmt       = "ai:agent:%d:mcp"
	agentSubagentKeyFmt  = "ai:agent:%d:subagents"
	agentPublishedKeyFmt = "ai:agent:%d:published"
	agentEnabledListKey  = "ai:agent:list:enabled"

	cacheInvalidationChannel = "cache:invalidation"
)

// invalidateCacheKeys 事务提交后删除缓存键并逐键广播失效（对齐 python CacheService.delete）。
func invalidateCacheKeys(ctx context.Context, keys ...string) {
	client := redis.GetClient()
	if client == nil || len(keys) == 0 {
		return
	}
	_ = client.Del(ctx, keys...).Err()
	for _, key := range keys {
		payload, err := json.Marshal(map[string]any{"type": "key", "key": key, "senderId": "dehaze-go"})
		if err != nil {
			continue
		}
		_ = client.Publish(ctx, cacheInvalidationChannel, payload).Err()
	}
}

// QueryTimePtr 解析可选时间参数（RFC3339 / "2006-01-02T15:04:05" / "2006-01-02 15:04:05" / "2006-01-02"）：
// 空串表示未传（nil, true）；非空但格式非法返回 false——python 侧为 datetime 类型参数，
// 非法格式报 A0400，此处不得静默丢弃该过滤条件（否则会返回未过滤结果，形成行为分叉）。
func QueryTimePtr(raw string) (*time.Time, bool) {
	if raw == "" {
		return nil, true
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05", "2006-01-02 15:04:05", "2006-01-02"} {
		if t, err := time.ParseInLocation(layout, raw, time.Local); err == nil {
			return &t, true
		}
	}
	return nil, false
}
