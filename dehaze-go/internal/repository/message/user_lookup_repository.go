package message

import (
	"context"
	"fmt"

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

func (r *UserLookupRepository) FindUserIDsByLevel(ctx context.Context, level int) ([]int64, error) {
	levelCode := fmt.Sprintf("level_%d", level)
	var ids []int64
	err := r.db.WithContext(ctx).
		Table("sys_member m").
		Joins("INNER JOIN sys_user u ON m.user_id = u.id").
		Where("m.level_code = ? AND m.deleted = 0 AND m.status = 1 AND u.deleted = 0 AND u.status = 1", levelCode).
		Pluck("m.user_id", &ids).Error
	return ids, err
}

func (r *UserLookupRepository) FindUserIDsByTag(ctx context.Context, tag string) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).
		Table("sys_user u").
		Joins("INNER JOIN sys_dept d ON u.dept_id = d.id").
		Where("u.deleted = 0 AND u.status = 1 AND d.deleted = 0 AND d.status = 1 AND d.name = ?", tag).
		Pluck("u.id", &ids).Error
	return ids, err
}

var _ IUserLookupRepository = (*UserLookupRepository)(nil)
