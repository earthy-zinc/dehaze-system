package audit_log

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"go.mongodb.org/mongo-driver/mongo"
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
