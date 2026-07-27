package message

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type NotificationSettingRepository struct {
	db *gorm.DB
}

func NewNotificationSettingRepository(db *gorm.DB) *NotificationSettingRepository {
	return &NotificationSettingRepository{db: db}
}

func (r *NotificationSettingRepository) FindByUserID(ctx context.Context, userID int64) (*model.SysNotificationSetting, error) {
	var s model.SysNotificationSetting
	err := r.db.WithContext(ctx).Where("user_id = ?", userID).First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func (r *NotificationSettingRepository) Create(ctx context.Context, setting *model.SysNotificationSetting) error {
	return r.db.WithContext(ctx).Create(setting).Error
}

func (r *NotificationSettingRepository) Update(ctx context.Context, setting *model.SysNotificationSetting) error {
	return r.db.WithContext(ctx).Model(setting).
		Select("push_enabled", "dnd_enabled", "dnd_start", "dnd_end", "preferences").
		Updates(setting).Error
}

var _ INotificationSettingRepository = (*NotificationSettingRepository)(nil)
