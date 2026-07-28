package login_log

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"go.mongodb.org/mongo-driver/mongo"
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
