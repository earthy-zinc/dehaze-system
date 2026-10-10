package dict

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/config"
)

// TestMain 把全局 Redis 客户端指向进程内 miniredis：字典写路径的失效包含对
// `dict:value:{typeCode}:*` 的 SCAN+DEL 与 cache:invalidation 广播（python 端据此清 L1），
// 不接管全局客户端就观察不到该行为。
func TestMain(m *testing.M) {
	mr, err := miniredis.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "启动 miniredis 失败: %v\n", err)
		os.Exit(1)
	}
	config.Config = &config.AppConfig{}
	config.Config.Cache.Redis.Addr = mr.Addr()
	if _, err := redis.InitRedis(); err != nil {
		fmt.Fprintf(os.Stderr, "初始化测试 Redis 失败: %v\n", err)
		mr.Close()
		os.Exit(1)
	}
	code := m.Run()
	mr.Close()
	os.Exit(code)
}

// TestClearDictCachesInvalidatesValuePrefixAndBroadcasts 字典写路径必须同时失效
// `dict:data:{typeCode}` 与 `dict:value:{typeCode}:*`，并就后者广播 pattern 失效消息。
//
// 事实源：python `app/service/dict_service.py` 的 `_defer_invalidate_dict_caches`
// （同时失效 DICT_OPTIONS_CACHE_PREFIX 与 DICT_VALUE_CACHE_PREFIX）；
// python 对端同款用例 tests/test_dict_cache_invalidation_contract.py。
func TestClearDictCachesInvalidatesValuePrefixAndBroadcasts(t *testing.T) {
	ctx := context.Background()
	client := redis.GetClient()
	require.NotNil(t, client)

	sub := client.Subscribe(ctx, cacheInvalidationChannel)
	t.Cleanup(func() { _ = sub.Close() })
	if _, err := sub.Receive(ctx); err != nil {
		t.Fatalf("订阅 %s 失败: %v", cacheInvalidationChannel, err)
	}
	msgs := sub.Channel()

	optionsKey := DictOptionsCachePrefix + "ai_model"
	valueKeys := []string{
		DictValueCachePrefix + "ai_model:default_model",
		DictValueCachePrefix + "ai_model:default_model.asr",
	}
	neighborKey := DictValueCachePrefix + "member_growth_rules:sign_in_value"
	for _, key := range append([]string{optionsKey}, append(append([]string{}, valueKeys...), neighborKey)...) {
		require.NoError(t, client.Set(ctx, key, "v", time.Hour).Err())
	}

	clearDictCaches(ctx, redis.NewRedisCache(client), "ai_model")

	for _, key := range append([]string{optionsKey}, valueKeys...) {
		n, err := client.Exists(ctx, key).Result()
		require.NoError(t, err)
		require.Zero(t, n, "缓存未失效（python 侧判定会滞后到 TTL 到期）: %s", key)
	}

	n, err := client.Exists(ctx, neighborKey).Result()
	require.NoError(t, err)
	require.EqualValues(t, 1, n, "失效范围必须按 typeCode 精确隔离，不得误删其它类型字典值")

	select {
	case msg := <-msgs:
		var payload map[string]any
		require.NoError(t, json.Unmarshal([]byte(msg.Payload), &payload))
		require.Equal(t, "pattern", payload["type"], "dict:value 为通配失效，必须发 pattern 消息")
		require.Equal(t, DictValueCachePrefix+"ai_model:*", payload["key"])
		require.Equal(t, "dehaze-go", payload["senderId"])
	case <-time.After(3 * time.Second):
		t.Fatal("未收到 dict:value 缓存失效广播（python 进程内 L1 仍持旧值）")
	}
}

// TestDictCachePrefixesMatchPythonContract 键形态即跨端契约：与 python
// `dict_service.DICT_OPTIONS_CACHE_PREFIX / DICT_VALUE_CACHE_PREFIX` 逐字一致。
func TestDictCachePrefixesMatchPythonContract(t *testing.T) {
	require.Equal(t, "dict:data:", DictOptionsCachePrefix)
	require.Equal(t, "dict:value:", DictValueCachePrefix)
}
