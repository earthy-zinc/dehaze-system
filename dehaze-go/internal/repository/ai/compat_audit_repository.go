package ai

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// compatCallCollection AI 兼容 API 调用审计集合（与 dehaze-python
// AiApiCallLogDocument.COLLECTION 一致，TTL 30 天自动过期，只追加不更新）
const compatCallCollection = "ai_api_call_log"

// CompatAuditRepository AI 兼容端点调用审计查询（MongoDB）
type CompatAuditRepository struct {
	collection *mongo.Collection
}

func NewCompatAuditRepository(db *mongo.Database) *CompatAuditRepository {
	return &CompatAuditRepository{collection: db.Collection(compatCallCollection)}
}

// CompatCallRecord 审计记录（字段名为 python 写入的 snake_case）
type CompatCallRecord struct {
	ID             string     `bson:"_id" json:"-"`
	UserID         *int64     `bson:"user_id" json:"-"`
	KeyID          *int64     `bson:"key_id" json:"keyId"`
	KeyPrefix      string     `bson:"key_prefix" json:"keyPrefix"`
	ConversationID *int64     `bson:"conversation_id" json:"conversationId"`
	Model          *string    `bson:"model" json:"model"`
	Endpoint       string     `bson:"endpoint" json:"endpoint"`
	Protocol       string     `bson:"protocol" json:"protocol"`
	IsStream       bool       `bson:"is_stream" json:"isStream"`
	InputTokens    int64      `bson:"input_tokens" json:"inputTokens"`
	OutputTokens   int64      `bson:"output_tokens" json:"outputTokens"`
	Credits        *float64   `bson:"credits" json:"credits"`
	StatusCode     int        `bson:"status_code" json:"statusCode"`
	DurationMs     int64      `bson:"duration_ms" json:"durationMs"`
	ClientIp       string     `bson:"client_ip" json:"clientIp"`
	RequestID      string     `bson:"request_id" json:"requestId"`
	ErrorMsg       *string    `bson:"error_msg" json:"errorMsg"`
	CreateTime     *time.Time `bson:"create_time" json:"createTime"`
}

// Query 按用户/Key/模型/时间筛选调用日志（create_time 倒序分页，user_id 强制过滤）
func (r *CompatAuditRepository) Query(
	ctx context.Context,
	userID int64,
	keyID *int64,
	model string,
	startTime, endTime *time.Time,
	page, size int,
) ([]CompatCallRecord, int64, error) {
	filter := bson.M{"user_id": userID}
	if keyID != nil {
		filter["key_id"] = *keyID
	}
	if model != "" {
		filter["model"] = model
	}
	timeRange := bson.M{}
	if startTime != nil {
		timeRange["$gte"] = *startTime
	}
	if endTime != nil {
		timeRange["$lte"] = *endTime
	}
	if len(timeRange) > 0 {
		filter["create_time"] = timeRange
	}

	total, err := r.collection.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	opts := options.Find().
		SetSort(bson.D{{Key: "create_time", Value: -1}}).
		SetSkip(int64((page - 1) * size)).
		SetLimit(int64(size))
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		return nil, 0, err
	}
	defer cursor.Close(ctx)

	records := make([]CompatCallRecord, 0)
	if err := cursor.All(ctx, &records); err != nil {
		return nil, 0, err
	}
	return records, total, nil
}
