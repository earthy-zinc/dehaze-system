package ai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// 三端共享的 AI 域缓存必须按 python 写入格式互认：
//   - ai:model:list    snake_case（python AiModelResult.model_dump(mode="json")）
//   - ai:provider:list snake_case（python ProviderEnabledResult.model_dump(mode="json")）
//   - ai:bill:{uid}:{month} snake_case（见 billing_test.go TestBillCacheCrossEndInterop）
//
// java 侧同样走 AiJsonUtils（SNAKE_MAPPER）。**自写自读无法暴露格式断裂**，
// 故此处一律用"另一端写入的夹具 → 本端读出"的方向断言。

// python 形态夹具：AiModelResult + speed_tier + is_fallback_target（snake_case，无 create_time）
const pythonModelListFixture = `[{"id":9001,"provider_id":7,"model_id":"cache-fixture-model",` +
	`"model_type":"chat","dimension":null,"display_name":"夹具模型",` +
	`"max_context_tokens":8192,"max_output_tokens":2048,"supports_multimodal":1,` +
	`"supports_tool_call":1,"supports_streaming":1,"supports_prompt_cache":1,` +
	`"supports_structured_output":0,"extra_request_params":{"enable_thinking":true},` +
	`"fallback_model_id":null,"prompt_cache_prefix_len":64,"status":1,"vip_level":2,` +
	`"last_test_status":1,"last_test_at":null,"last_test_error":null,"calls_24h":12,` +
	`"success_rate_24h":99,"last_call_at":null,"speed_tier":"fast","is_fallback_target":true}]`

// java 形态夹具：AiJsonUtils + AiBillVO/ProviderEnabledVO（snake_case，status 为数字）
const pythonProviderListFixture = `[{"id":8001,"provider_code":"cache-fixture-provider",` +
	`"display_name":"夹具供应商","protocol_type":"openai_compat","health":"healthy","status":1}]`

func TestModelListCacheReadsPeerFixture(t *testing.T) {
	db := testutil.NewTestDB(t)
	client := initTestRedisClient(t)
	ctx := context.Background()
	resetCacheKey(t, client, modelListCacheKey)

	service := NewModelService(db, airepo.NewModelRepository(db),
		airepo.NewModelPriceRepository(db), NewHealthService(db), nil)

	require.NoError(t, client.Set(ctx, modelListCacheKey, pythonModelListFixture, 0).Err())

	items, err := service.loadEnabledModelCache(ctx)
	require.NoError(t, err)
	require.Len(t, items, 1, "必须直接读出对端写的缓存（回落 DB 则不会出现夹具模型）")

	item := items[0]
	require.EqualValues(t, 9001, item.ID)
	require.EqualValues(t, 7, item.ProviderID)
	require.Equal(t, "cache-fixture-model", item.ModelID)
	require.Equal(t, "夹具模型", item.DisplayName)
	require.Equal(t, "chat", item.ModelType)
	require.EqualValues(t, 8192, item.MaxContextTokens)
	require.EqualValues(t, 1, item.SupportsMultimodal)
	require.JSONEq(t, `{"enable_thinking":true}`, string(item.ExtraRequestParams))
	require.EqualValues(t, 2, item.VipLevel)
	require.EqualValues(t, 12, *item.Calls24h)
	require.EqualValues(t, 99, *item.SuccessRate24h)
	require.Equal(t, "fast", item.SpeedTier)
	require.True(t, item.IsFallbackTarget)
	require.Nil(t, item.FallbackModelID)
}

func TestModelListCacheWritesSnakeCase(t *testing.T) {
	models := []modelCacheDTO{}
	require.NoError(t, json.Unmarshal([]byte(pythonModelListFixture), &models))
	require.Len(t, models, 1)

	roundTripped := modelFromCache(&models[0])
	payload, err := json.Marshal([]modelCacheDTO{modelToCache(&roundTripped)})
	require.NoError(t, err)
	text := string(payload)

	require.Contains(t, text, `"provider_id":7`)
	require.Contains(t, text, `"model_id":"cache-fixture-model"`)
	require.Contains(t, text, `"speed_tier":"fast"`)
	require.Contains(t, text, `"is_fallback_target":true`)
	require.Contains(t, text, `"calls_24h":12`)
	require.NotContains(t, text, `"providerId"`, "共享缓存严禁出现 camelCase 键")
	require.NotContains(t, text, `"modelId"`)
	require.NotContains(t, text, `"speedTier"`)
}

func TestProviderListCacheReadsPeerFixture(t *testing.T) {
	client := initTestRedisClient(t)
	ctx := context.Background()
	resetCacheKey(t, client, providerListCacheKey)

	service := NewProviderService(airepo.NewProviderRepository(nil),
		airepo.NewProviderKeyRepository(nil), NewHealthService(nil))

	require.NoError(t, client.Set(ctx, providerListCacheKey, pythonProviderListFixture, 0).Err())
	items, err := service.ListEnabledProviders(ctx)
	require.NoError(t, err)
	require.Len(t, items, 1, "必须直接读出对端写的缓存")
	require.EqualValues(t, 8001, items[0].ID)
	require.Equal(t, "cache-fixture-provider", items[0].ProviderCode)
	require.Equal(t, "夹具供应商", items[0].DisplayName)
	require.Equal(t, "openai_compat", items[0].ProtocolType)
	require.Equal(t, "healthy", *items[0].Health)
	require.EqualValues(t, 1, items[0].Status)
}

func TestProviderListCacheWritesSnakeCase(t *testing.T) {
	health := "healthy"
	items := []vo.ProviderEnabledVO{{ID: 8001, ProviderCode: "p", DisplayName: "供应商",
		ProtocolType: "openai_compat", Health: &health, Status: 1}}
	cachePayload := make([]providerCacheDTO, 0, len(items))
	for i := range items {
		cachePayload = append(cachePayload, providerToCache(&items[i]))
	}
	payload, err := json.Marshal(cachePayload)
	require.NoError(t, err)
	text := string(payload)

	require.Contains(t, text, `"provider_code":"p"`)
	require.Contains(t, text, `"display_name":"供应商"`)
	require.Contains(t, text, `"protocol_type":"openai_compat"`)
	require.NotContains(t, text, `"providerCode"`)
	require.NotContains(t, text, `"displayName"`)
	require.NotContains(t, text, `"protocolType"`)
}

// initTestRedisClient 初始化 pkg/cache/redis 全局客户端（config.test.yaml，db=4 与开发库隔离），
// 使 service 内的 redisClient() 可用；用例自管键清理，不污染其他用例。
func initTestRedisClient(t *testing.T) *goredis.Client {
	t.Helper()
	testutil.LoadTestConfig(t)
	if _, err := redis.InitRedis(); err != nil {
		t.Fatalf("测试 Redis 初始化失败（config.test.yaml db=4）: %v", err)
	}
	client := redis.GetClient()
	require.NotNil(t, client, "Redis 客户端不可用")
	return client
}

func resetCacheKey(t *testing.T, client *goredis.Client, key string) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, client.Del(ctx, key).Err())
	t.Cleanup(func() { _ = client.Del(context.Background(), key).Err() })
}
