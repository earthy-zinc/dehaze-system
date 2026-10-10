package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// UserPreferenceRepository 用户 AI 偏好设置数据访问（sys_ai_user_preference）
type UserPreferenceRepository struct {
	db *gorm.DB
}

func NewUserPreferenceRepository(db *gorm.DB) *UserPreferenceRepository {
	return &UserPreferenceRepository{db: db}
}

// FindByUserID 查询用户偏好；无记录返回 nil（不建行，与 GET 契约一致）
func (r *UserPreferenceRepository) FindByUserID(ctx context.Context, userID int64) (*model.SysAiUserPreference, error) {
	var pref model.SysAiUserPreference
	err := r.db.WithContext(ctx).Where("user_id = ? AND deleted = 0", userID).First(&pref).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &pref, nil
}

// Upsert 写入偏好：命中唯一键 uk_user_id(user_id, deleted) 时只更新 default_model/update_by
func (r *UserPreferenceRepository) Upsert(ctx context.Context, pref *model.SysAiUserPreference) error {
	return r.db.WithContext(ctx).Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "user_id"}, {Name: "deleted"}},
		DoUpdates: clause.Assignments(map[string]any{
			"default_model": pref.DefaultModel,
			"update_by":     pref.UpdateBy,
			"update_time":   time.Now(),
		}),
	}).Create(pref).Error
}
