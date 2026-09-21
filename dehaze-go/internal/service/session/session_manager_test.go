package session

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	cacheredis "github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

// 进程级 miniredis：会话索引（ZSet）逻辑依赖真实 Redis 命令语义，不用 mock 缓存自证。
var testRedis *miniredis.Miniredis

func TestMain(m *testing.M) {
	mr, err := miniredis.Run()
	if err != nil {
		panic(err)
	}
	testRedis = mr
	config.Config = &config.AppConfig{
		Cache: options.Cache{
			Type:  "redis",
			Redis: options.Redis{Enabled: true, Addr: mr.Addr(), DB: 0},
		},
	}
	if _, err := cache.Init(); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func sessionIndexKey(userID int64) string {
	return common.SessionUserPrefix + strconv.FormatInt(userID, 10)
}

func indexMembers(t *testing.T, userID int64) []string {
	t.Helper()
	members, err := testRedis.ZMembers(sessionIndexKey(userID))
	assert.NoError(t, err)
	return members
}

// seedSession 写入会话键并将其登记进索引（score 显式指定，模拟历史登录顺序）。
func seedSession(t *testing.T, sessionID string, userID int64, score float64) {
	t.Helper()
	client := cacheredis.GetClient()
	payload, err := json.Marshal(middleware.SessionData{
		UserID:   userID,
		Username: "u" + strconv.FormatInt(userID, 10),
	})
	assert.NoError(t, err)
	assert.NoError(t, client.Set(context.Background(), common.SessionPrefix+sessionID, payload, middleware.SessionTTL).Err())
	assert.NoError(t, client.ZAdd(context.Background(), sessionIndexKey(userID), goredis.Z{Score: score, Member: sessionID}).Err())
}

func TestRegisterSession_EvictsEarliest_WhenOverLimit(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7001)
	base := float64(time.Now().Unix() - 3600)

	seedSession(t, "s-oldest", userID, base)
	seedSession(t, "s-middle", userID, base+1)
	seedSession(t, "s-newest", userID, base+2)

	// 上限 3：第 4 台登录踢掉最早登录者
	assert.NoError(t, RegisterSession(ctx, userID, "s-brand-new", 3))

	members := indexMembers(t, userID)
	assert.Len(t, members, 3)
	assert.NotContains(t, members, "s-oldest")
	assert.Contains(t, members, "s-middle")
	assert.Contains(t, members, "s-newest")
	assert.Contains(t, members, "s-brand-new")

	// 被踢会话键已删除（下一请求 401），其余会话不受影响
	_, err := testRedis.Get(common.SessionPrefix + "s-oldest")
	assert.Error(t, err)
	_, err = testRedis.Get(common.SessionPrefix + "s-middle")
	assert.NoError(t, err)
	_, err = testRedis.Get(common.SessionPrefix + "s-newest")
	assert.NoError(t, err)
}

func TestRegisterSession_KeepsAll_WhenWithinLimit(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7002)
	base := float64(time.Now().Unix() - 3600)

	for idx := 0; idx < 9; idx++ {
		seedSession(t, "s-"+strconv.Itoa(idx), userID, base+float64(idx))
	}

	assert.NoError(t, RegisterSession(ctx, userID, "s-new", AdminMaxDevices))

	assert.Len(t, indexMembers(t, userID), 10)
	for idx := 0; idx < 9; idx++ {
		_, err := testRedis.Get(common.SessionPrefix + "s-" + strconv.Itoa(idx))
		assert.NoError(t, err)
	}
}

func TestRegisterSession_OverflowEvictsExactlyExcess(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7003)
	base := float64(time.Now().Unix() - 3600)

	for idx := 0; idx < 7; idx++ {
		seedSession(t, "s-"+strconv.Itoa(idx), userID, base+float64(idx))
	}

	// 上限 5：8 台在线只踢最早的 3 台
	assert.NoError(t, RegisterSession(ctx, userID, "s-new", 5))

	assert.Len(t, indexMembers(t, userID), 5)
	for idx := 0; idx < 3; idx++ {
		_, err := testRedis.Get(common.SessionPrefix + "s-" + strconv.Itoa(idx))
		assert.Error(t, err)
	}
	_, err := testRedis.Get(common.SessionPrefix + "s-6")
	assert.NoError(t, err)
}

// 本项目时间列全为 DATETIME 秒精度，索引 score 若带亚秒会把三端排序口径拉开。
func TestRegisterSession_ScoreTruncatedToSecond(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7004)

	before := time.Now().Unix()
	assert.NoError(t, RegisterSession(ctx, userID, "s-score", 1))
	after := time.Now().Unix()

	score, err := testRedis.ZScore(sessionIndexKey(userID), "s-score")
	assert.NoError(t, err)
	assert.Equal(t, score, float64(int64(score)), "score 必须为整秒")
	assert.GreaterOrEqual(t, int64(score), before)
	assert.LessOrEqual(t, int64(score), after)
}

func TestRegisterSession_IndexHasSessionTTL(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7005)

	assert.NoError(t, RegisterSession(ctx, userID, "s-ttl", 1))
	assert.Greater(t, testRedis.TTL(sessionIndexKey(userID)), 6*24*time.Hour)
}

func TestRemoveFromIndex_KeepsOtherMembers(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	userID := int64(7006)
	base := float64(time.Now().Unix() - 60)
	seedSession(t, "s-a", userID, base)
	seedSession(t, "s-b", userID, base+1)

	assert.NoError(t, RemoveFromIndex(ctx, userID, "s-a"))

	assert.Equal(t, []string{"s-b"}, indexMembers(t, userID))
	// 注销只剔除索引元素，会话键由调用方（Logout）删除
	_, err := testRedis.Get(common.SessionPrefix + "s-a")
	assert.NoError(t, err)
}

func TestKickByUserIDs_CleansSessionIndex(t *testing.T) {
	testRedis.FlushAll()
	ctx := context.Background()
	targetUser, otherUser := int64(7007), int64(7008)
	base := float64(time.Now().Unix() - 60)
	seedSession(t, "s-t1", targetUser, base)
	seedSession(t, "s-t2", targetUser, base+1)
	seedSession(t, "s-o1", otherUser, base)

	kicked, err := KickByUserIDs(ctx, []int64{targetUser})

	assert.NoError(t, err)
	assert.Equal(t, 2, kicked)
	assert.False(t, testRedis.Exists(sessionIndexKey(targetUser)), "被踢用户的会话索引应清理")
	assert.True(t, testRedis.Exists(sessionIndexKey(otherUser)), "其他用户索引不受影响")
}
