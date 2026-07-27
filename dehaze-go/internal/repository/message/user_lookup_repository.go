package message

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type UserLookupRepository struct {
	db *gorm.DB
}

func NewUserLookupRepository(db *gorm.DB) *UserLookupRepository {
	return &UserLookupRepository{db: db}
}

func (r *UserLookupRepository) FindAllUserIDs(ctx context.Context) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Where("deleted = 0 AND status = 1").
		Pluck("id", &ids).Error
	return ids, err
}

var _ IUserLookupRepository = (*UserLookupRepository)(nil)
