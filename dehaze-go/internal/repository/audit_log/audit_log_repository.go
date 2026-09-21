package audit_log

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type AuditLogRepository struct {
	collection *mongo.Collection
}

func NewAuditLogRepository(db *mongo.Database) *AuditLogRepository {
	return &AuditLogRepository{collection: db.Collection(model.AuditLog{}.CollectionName())}
}

func (r *AuditLogRepository) Create(ctx context.Context, log *model.AuditLog) error {
	_, err := r.collection.InsertOne(ctx, log)
	return err
}

// ListByTarget 按目标对象分页查询审计日志（create_time 倒序），返回 (列表, 总数)。
// 过滤条件与排序对齐 python `mongo_audit_log_repository.list_by_target`。
//
// target_id 用 int32/int64 双类型匹配：python(pymongo) 把小于 2^31 的整数编码为 **int32**，
// go 用 int64 查询会取不到 python 写入的日志（BSON 数值类型不相等），故两种都纳入。
func (r *AuditLogRepository) ListByTarget(
	ctx context.Context, targetType string, targetID int64, page, pageSize int,
) ([]model.AuditLog, int64, error) {
	filter := bson.M{
		"target_type": targetType,
		"target_id":   bson.M{"$in": bson.A{int32(targetID), targetID}},
	}

	total, err := r.collection.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}

	cursor, err := r.collection.Find(ctx, filter, options.Find().
		SetSort(bson.D{{Key: "create_time", Value: -1}}).
		SetSkip(int64((page-1)*pageSize)).
		SetLimit(int64(pageSize)))
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = cursor.Close(ctx) }()

	var items []model.AuditLog
	if err := cursor.All(ctx, &items); err != nil {
		return nil, 0, err
	}
	return items, total, nil
}
