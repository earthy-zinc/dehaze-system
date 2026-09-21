package aidomain

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// keyUnavailableKeyPattern 与 python provider_key_selector.KEY_UNAVAILABLE_PREFIX 一致。
const keyUnavailableKeyPattern = "ai:provider_key:*:unavailable"

// providerHealthSnapshotKeyFmt 与 python provider_health_service 的缓存键一致。
const providerHealthSnapshotKeyFmt = "ai:provider:%d:health"

// ProviderHealthStatVO 供应商健康看板项。
type ProviderHealthStatVO struct {
	ProviderID   int64   `json:"providerId"`
	ProviderName string  `json:"providerName"`
	Health       string  `json:"health"`
	CallCount    int64   `json:"callCount"`
	SuccessRate  float64 `json:"successRate"`
	Rate429      float64 `json:"rate429"`
	P95LatencyMs int64   `json:"p95LatencyMs"`
	CircuitOpen  bool    `json:"circuitOpen"`
}

// ModelUsageStatVO 模型用量分布项。
type ModelUsageStatVO struct {
	ModelID      string `json:"modelId"`
	DisplayName  string `json:"displayName"`
	CallCount    int64  `json:"callCount"`
	InputTokens  int64  `json:"inputTokens"`
	OutputTokens int64  `json:"outputTokens"`
	Credits      int64  `json:"credits"`
}

// DowngradeStatVO 降级频率项。
type DowngradeStatVO struct {
	ModelID string `json:"modelId"`
	Count   int64  `json:"count"`
}

// DegradeFaultStatVO 降级与故障统计。
type DegradeFaultStatVO struct {
	DowngradeFrequency []DowngradeStatVO `json:"downgradeFrequency"`
	KeyFailoverCount   int64             `json:"keyFailoverCount"`
}

// UsageStatsVO 运营统计结果。
type UsageStatsVO struct {
	ProviderHealth []ProviderHealthStatVO `json:"providerHealth"`
	ModelUsage     []ModelUsageStatVO     `json:"modelUsage"`
	DegradeFault   DegradeFaultStatVO     `json:"degradeFault"`
}

// UsageService 运营统计（供应商健康/模型用量/降级与故障）。
type UsageService struct {
	usage *repo.UsageRepository
}

// NewUsageService 构造 UsageService。
func NewUsageService(usage *repo.UsageRepository) *UsageService {
	return &UsageService{usage: usage}
}

// GetUsageStats 读取运营统计。
func (s *UsageService) GetUsageStats(ctx context.Context, startTime, endTime *time.Time) (*UsageStatsVO, error) {
	providerHealth, err := s.providerHealth(ctx)
	if err != nil {
		return nil, err
	}
	modelUsage, err := s.modelUsage(ctx, startTime, endTime)
	if err != nil {
		return nil, err
	}
	degradeFault, err := s.degradeFault(ctx, startTime, endTime)
	if err != nil {
		return nil, err
	}
	return &UsageStatsVO{
		ProviderHealth: providerHealth,
		ModelUsage:     modelUsage,
		DegradeFault:   degradeFault,
	}, nil
}

// providerHealth 供应商健康看板：读 python 写入的 Redis 健康快照（无快照按健康默认值展示）。
func (s *UsageService) providerHealth(ctx context.Context) ([]ProviderHealthStatVO, error) {
	providers, err := s.usage.ListProviders(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询供应商失败", err)
	}
	client := redis.GetClient()
	items := make([]ProviderHealthStatVO, 0, len(providers))
	for _, provider := range providers {
		item := ProviderHealthStatVO{
			ProviderID:   provider.ID,
			ProviderName: provider.DisplayName,
			Health:       "healthy",
		}
		if client != nil {
			if raw, err := client.Get(ctx, fmt.Sprintf(providerHealthSnapshotKeyFmt, provider.ID)).Bytes(); err == nil && len(raw) > 0 {
				snapshot := map[string]any{}
				if json.Unmarshal(raw, &snapshot) == nil {
					if status, ok := snapshot["status"].(string); ok {
						item.Health = status
					}
					if value, ok := toFloat(snapshot["total_calls_24h"]); ok {
						item.CallCount = int64(value)
					}
					if value, ok := toFloat(snapshot["success_rate"]); ok {
						item.SuccessRate = value
					}
					if value, ok := toFloat(snapshot["limit_rate"]); ok {
						item.Rate429 = value
					}
					if value, ok := toFloat(snapshot["p95_latency_ms"]); ok {
						item.P95LatencyMs = int64(value)
					}
					if value, ok := snapshot["circuit_open"].(bool); ok {
						item.CircuitOpen = value
					}
				}
			}
		}
		items = append(items, item)
	}
	return items, nil
}

// modelUsage 按模型聚合计费流水。
func (s *UsageService) modelUsage(ctx context.Context, startTime, endTime *time.Time) ([]ModelUsageStatVO, error) {
	rows, err := s.usage.ModelUsageByModel(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "模型用量聚合失败", err)
	}
	modelIDs := make([]string, 0, len(rows))
	for _, row := range rows {
		modelIDs = append(modelIDs, row.Model)
	}
	displayNames, err := s.usage.ModelDisplayNames(ctx, modelIDs)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询模型名称失败", err)
	}
	items := make([]ModelUsageStatVO, 0, len(rows))
	for _, row := range rows {
		displayName := displayNames[row.Model]
		if displayName == "" {
			displayName = row.Model
		}
		items = append(items, ModelUsageStatVO{
			ModelID:      row.Model,
			DisplayName:  displayName,
			CallCount:    row.CallCount,
			InputTokens:  row.InputTokens,
			OutputTokens: row.OutputTokens,
			Credits:      row.Credits,
		})
	}
	return items, nil
}

// degradeFault 降级频率（actual_model 非空即发生降级）+ 冷却期 Key 数。
func (s *UsageService) degradeFault(ctx context.Context, startTime, endTime *time.Time) (DegradeFaultStatVO, error) {
	rows, err := s.usage.DowngradeByModel(ctx, startTime, endTime)
	if err != nil {
		return DegradeFaultStatVO{}, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "降级统计失败", err)
	}
	frequency := make([]DowngradeStatVO, 0, len(rows))
	for _, row := range rows {
		if row.Count > 0 {
			frequency = append(frequency, DowngradeStatVO{ModelID: row.ModelID, Count: row.Count})
		}
	}
	return DegradeFaultStatVO{
		DowngradeFrequency: frequency,
		KeyFailoverCount:   countUnavailableKeys(ctx),
	}, nil
}

// countUnavailableKeys 统计处于冷却期的 Key 数。
func countUnavailableKeys(ctx context.Context) int64 {
	client := redis.GetClient()
	if client == nil {
		return 0
	}
	var cursor uint64
	var count int64
	for {
		keys, next, err := client.Scan(ctx, cursor, keyUnavailableKeyPattern, 100).Result()
		if err != nil {
			return count
		}
		count += int64(len(keys))
		cursor = next
		if cursor == 0 {
			return count
		}
	}
}
