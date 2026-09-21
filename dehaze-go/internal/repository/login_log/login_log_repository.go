package login_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type LoginLogRepository struct {
	collection *mongo.Collection
}

func NewLoginLogRepository(db *mongo.Database) *LoginLogRepository {
	return &LoginLogRepository{collection: db.Collection(model.LoginLog{}.CollectionName())}
}

func (r *LoginLogRepository) Create(ctx context.Context, log *model.LoginLog) error {
	_, err := r.collection.InsertOne(ctx, log)
	return err
}

// PageLogs 分页查询登录日志（create_time 倒序），查询条件与 Python 端 page_logs 对齐：
// username/ip/device_type/status 精确匹配，create_time 范围限定，user_ids 限定可见用户范围。
func (r *LoginLogRepository) PageLogs(ctx context.Context, q *query.LoginLogQuery) ([]model.LoginLog, int64, error) {
	filter := r.buildFilter(q)

	total, err := r.collection.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}

	opts := options.Find().
		SetSort(bson.D{{Key: "create_time", Value: -1}}).
		SetSkip(int64((q.PageNum - 1) * q.PageSize)).
		SetLimit(int64(q.PageSize))
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		return nil, 0, err
	}
	defer cursor.Close(ctx)

	logs := make([]model.LoginLog, 0)
	if err := cursor.All(ctx, &logs); err != nil {
		return nil, 0, err
	}
	return logs, total, nil
}

func (r *LoginLogRepository) buildFilter(q *query.LoginLogQuery) bson.M {
	filter := bson.M{}
	if q.Username != "" {
		filter["username"] = q.Username
	}
	if q.IP != "" {
		filter["ip"] = q.IP
	}
	if q.Status != nil {
		filter["status"] = *q.Status
	}
	if q.DeviceType != "" {
		filter["device_type"] = q.DeviceType
	}
	if len(q.UserIDs) > 0 {
		filter["user_id"] = bson.M{"$in": q.UserIDs}
	}
	timeRange := bson.M{}
	if start, ok := parseLogTime(q.StartTime); ok {
		timeRange["$gte"] = start
	}
	if end, ok := parseLogTime(q.EndTime); ok {
		timeRange["$lte"] = end
	}
	if len(timeRange) > 0 {
		filter["create_time"] = timeRange
	}
	return filter
}

// parseLogTime 支持 "2006-01-02 15:04:05"、"2006-01-02T15:04:05"、"2006-01-02" 三种格式，与 Python 端 _parse_log_time 对齐
func parseLogTime(value string) (time.Time, bool) {
	if value == "" {
		return time.Time{}, false
	}
	layouts := []string{"2006-01-02 15:04:05", "2006-01-02T15:04:05", "2006-01-02"}
	for _, layout := range layouts {
		if t, err := time.ParseInLocation(layout, value, time.Local); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}
