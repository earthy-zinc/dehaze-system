package audit_log

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/mongo"
	"github.com/stretchr/testify/require"
	"go.mongodb.org/mongo-driver/bson"
)

// TestListByTargetReadsPythonFormDocument 跨端互认（与 internal/service/ai/cache_interop_test.go 同方法学）：
// 真存储 + **python 形态夹具** → 本端读出断言，证明读侧真能读到对端写入。
//
// 夹具刻意用 python 的写法：键名 snake_case、`target_id` 用 **int32**（pymongo 对 <2^31 整数的默认编码，
// go 若用 int64 查会取不到）、`create_time` 为 UTC Date。若 go 侧 bson 标签仍是 camelCase，
// 本用例会读到全零值/查不到 —— 自写自读无法暴露这种格式断裂。
func TestListByTargetReadsPythonFormDocument(t *testing.T) {
	testutil.LoadTestConfig(t)
	require.NoError(t, mongo.InitMongo(), "测试环境需要 MongoDB（config.test.yaml 的 mongo.uri）")

	db := mongo.GetMongoDatabase("")
	require.NotNil(t, db, "Mongo 未初始化")

	repo := NewAuditLogRepository(db)
	collection := db.Collection(model.AuditLog{}.CollectionName())
	ctx := context.Background()

	targetID := int64(790001)
	pythonFilter := bson.M{
		"target_type": "member",
		"target_id":   bson.M{"$in": bson.A{int32(targetID), targetID}},
	}
	_, err := collection.DeleteMany(ctx, pythonFilter)
	require.NoError(t, err)
	t.Cleanup(func() { _, _ = collection.DeleteMany(ctx, pythonFilter) })

	createTime := time.Date(2026, 9, 18, 3, 4, 5, 0, time.UTC)
	_, err = collection.InsertOne(ctx, bson.M{
		"operator_id":  int64(2),
		"target_type":  "member",
		"target_id":    int32(targetID),
		"action":       "level_change",
		"module":       "member",
		"before_value": "level_0",
		"after_value":  "level_1",
		"ip":           "127.0.0.1",
		"user_agent":   "python-httpx/0.27",
		"create_time":  createTime,
	})
	require.NoError(t, err)

	// 负向对照（用例自证）：camelCase 口径（go 旧标签写法）必须查不到 python 写入的文档，
	// 否则说明夹具本身混入了 camelCase，用例的"格式断裂"捕获能力是假的。
	camelCount, err := collection.CountDocuments(ctx, bson.M{"targetType": "member", "targetId": int32(targetID)})
	require.NoError(t, err)
	require.EqualValues(t, 0, camelCount, "python 形态文档只有 snake_case 键")

	items, total, err := repo.ListByTarget(ctx, "member", targetID, 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, total, "int32 形态的 target_id 必须能被 int64 查询命中")
	require.Len(t, items, 1)

	item := items[0]
	require.NotEmpty(t, item.ID.Hex(), "_id 必须回填（响应 id 用）")
	require.EqualValues(t, 2, item.OperatorID)
	require.Equal(t, "level_change", item.Action)
	require.Equal(t, "member", item.Module)
	require.Equal(t, "level_0", item.BeforeValue)
	require.Equal(t, "level_1", item.AfterValue)
	require.Equal(t, "127.0.0.1", item.IP)
	require.Equal(t, "python-httpx/0.27", item.UserAgent)
	require.WithinDuration(t, createTime, item.CreateTime, time.Second)
}
