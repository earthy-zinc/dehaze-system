package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/config"
)

// TestMain 把全局 Redis 客户端指向进程内 miniredis：invalidateReasoningGraphs 经全局客户端
// 发布 cache:invalidation 广播，不接管全局客户端就观察不到该行为。
func TestMain(m *testing.M) {
	cache, err := miniredis.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "启动 miniredis 失败: %v\n", err)
		os.Exit(1)
	}
	config.Config = &config.AppConfig{}
	config.Config.Cache.Redis.Addr = cache.Addr()
	if _, err := redis.InitRedis(); err != nil {
		fmt.Fprintf(os.Stderr, "初始化测试 Redis 失败: %v\n", err)
		cache.Close()
		os.Exit(1)
	}
	code := m.Run()
	cache.Close()
	os.Exit(code)
}

// subscribeInvalidation 订阅失效频道并等待订阅确认，返回消息通道
func subscribeInvalidation(t *testing.T, ctx context.Context) <-chan *goredis.Message {
	t.Helper()
	client := redis.GetClient()
	require.NotNil(t, client)
	sub := client.Subscribe(ctx, cacheInvalidationChannel)
	t.Cleanup(func() { _ = sub.Close() })
	if _, err := sub.Receive(ctx); err != nil {
		t.Fatalf("订阅 %s 失败: %v", cacheInvalidationChannel, err)
	}
	return sub.Channel()
}

// requireGraphInvalidationBroadcast 校验广播载荷即跨端协议：{type, senderId} 且无 key
// （python 订阅端按 type 分支失效进程内推理图缓存）。
func requireGraphInvalidationBroadcast(t *testing.T, ch <-chan *goredis.Message) {
	t.Helper()
	select {
	case msg := <-ch:
		var payload map[string]any
		require.NoError(t, json.Unmarshal([]byte(msg.Payload), &payload))
		require.Equal(t, "ai_graph_invalidate", payload["type"])
		require.Equal(t, "dehaze-go", payload["senderId"])
		require.NotContains(t, payload, "key")
	case <-time.After(3 * time.Second):
		t.Fatal("未收到推理图缓存失效广播（python 已构图仍会持旧 MCP 工具集）")
	}
}

func TestInvalidateReasoningGraphsPublishesProtocolPayload(t *testing.T) {
	ctx := context.Background()
	ch := subscribeInvalidation(t, ctx)

	invalidateReasoningGraphs(ctx)

	requireGraphInvalidationBroadcast(t, ch)
}

// TestUpdateNamespacesBroadcastsGraphInvalidation MCP 命名空间变更后必须广播推理图缓存失效：
// python 运行面已构建的图仍持旧工具集，Go 原生改 MCP 不会触发其重建，只能靠该消息失效。
func TestUpdateNamespacesBroadcastsGraphInvalidation(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	repo := airepo.NewMcpRepository(db)
	svc := NewMcpServerService(db, repo)
	ctx := context.Background()

	endpoint := "https://example.com/mcp"
	authType := "api_key"
	server := &model.SysAiMcpServer{
		Name:         fmt.Sprintf("mcp_invalidate_%d", time.Now().UnixNano()),
		ProtocolType: "streamable-http",
		Endpoint:     &endpoint,
		AuthType:     &authType,
		Status:       1,
	}
	require.NoError(t, repo.CreateServer(ctx, server))
	require.NoError(t, db.Exec(
		"INSERT INTO sys_ai_mcp_tool (server_id, name) VALUES (?, ?)",
		server.ID, "search_web").Error)

	ch := subscribeInvalidation(t, ctx)

	_, err := svc.UpdateNamespaces(ctx, server.ID,
		[]bo.McpNamespaceForm{{Name: "web", ToolNames: []string{"search_web"}}}, 1)
	require.NoError(t, err)

	requireGraphInvalidationBroadcast(t, ch)
}
