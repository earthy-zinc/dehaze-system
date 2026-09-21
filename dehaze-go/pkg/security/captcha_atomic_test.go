package security

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// cmdRecorder 记录实际下发的 Redis 命令名。用于**自证**消费走的是 GETDEL：
// 并发断言只能证明"结果正确"，命令断言能证明"路径正确"（旧实现是 GET + DEL 两条命令）。
type cmdRecorder struct {
	mu   sync.Mutex
	cmds []string
}

func (h *cmdRecorder) DialHook(next goredis.DialHook) goredis.DialHook { return next }

func (h *cmdRecorder) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		h.mu.Lock()
		h.cmds = append(h.cmds, cmd.Name())
		h.mu.Unlock()
		return next(ctx, cmd)
	}
}

func (h *cmdRecorder) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error { return next(ctx, cmds) }
}

func (h *cmdRecorder) reset() {
	h.mu.Lock()
	h.cmds = nil
	h.mu.Unlock()
}

func (h *cmdRecorder) names() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.cmds...)
}

// TestCaptchaStoreConsumeIsAtomic 验证码消费必须**原子**（python `verify_captcha_status` 用 Redis GETDEL）：
// 并发消费同一 captchaKey 时恰好一个请求取到值，其余取空 —— 否则并发登录会双双成功，
// 即 SDK `auth-security.test.ts:157`「同一 captchaKey 并发提交登录仅一个成功」的失败根因。
func TestCaptchaStoreConsumeIsAtomic(t *testing.T) {
	testutil.LoadTestConfig(t)
	if _, err := redis.InitRedis(); err != nil {
		t.Fatalf("测试 Redis 初始化失败（config.test.yaml）: %v", err)
	}
	client := redis.GetClient()
	require.NotNil(t, client, "Redis 客户端不可用")

	// CacheStore 读写走包级 cacheClient（生产由 GetCaptchaStore 注入 cache.GetCache()）；
	// 这里显式接入 Redis 实现，保证测的是真实原子语义
	prev := cacheClient
	cacheClient = redis.NewRedisCache(client)
	t.Cleanup(func() { cacheClient = prev })

	store := NewCacheStore()
	key := fmt.Sprintf("atomic-%d", time.Now().UnixNano())
	fullKey := common.CaptchaCodePrefix + key
	_ = client.Del(context.Background(), fullKey)
	t.Cleanup(func() { _ = client.Del(context.Background(), fullKey) })

	require.NoError(t, store.Set(key, "1234"))

	// 命令级自证：消费必须只下发一条 GETDEL（旧实现是 GET + DEL 两条，非原子）
	rec := &cmdRecorder{}
	client.AddHook(rec)
	rec.reset()
	require.Equal(t, "1234", store.Get(key, true))
	require.Equal(t, []string{"getdel"}, rec.names(), "消费必须走 GETDEL 单命令原子路径")
	require.Equal(t, "", store.Get(key, true), "已消费的验证码不得再次取到")

	require.NoError(t, store.Set(key, "1234"))

	const goroutines = 8
	var wg sync.WaitGroup
	results := make([]string, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = store.Get(key, true)
		}(i)
	}
	wg.Wait()

	consumed := 0
	for _, r := range results {
		if r != "" {
			consumed++
			require.Equal(t, "1234", r)
		}
	}
	require.Equal(t, 1, consumed, "并发消费同一验证码只允许一个请求成功")

	// clear=false 的校验读不得消费凭证（并发登录的失败方按 A0213 判"验证码已被消费"）
	require.NoError(t, store.Set(key, "5678"))
	require.Equal(t, "5678", store.Get(key, false))
	require.Equal(t, "5678", store.Get(key, false), "clear=false 的读不得删除验证码")
}
