package message

import (
	"context"
	"encoding/json"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

const dndTimeFormat = "15:04:05"

type NotificationSettingService struct {
	settingRepo msgrepo.INotificationSettingRepository
}

func NewNotificationSettingService(settingRepo msgrepo.INotificationSettingRepository) *NotificationSettingService {
	return &NotificationSettingService{settingRepo: settingRepo}
}

func (s *NotificationSettingService) Get(ctx context.Context, userID int64) (*vo.NotificationSettingsVO, error) {
	setting, err := s.getOrCreateDefault(ctx, userID)
	if err != nil {
		return nil, err
	}
	return s.toVO(setting), nil
}

func (s *NotificationSettingService) Update(ctx context.Context, userID int64, form *bo.NotificationSettingForm) error {
	setting, err := s.getOrCreateDefault(ctx, userID)
	if err != nil {
		return err
	}

	if form.PushEnabled != nil {
		setting.PushEnabled = boolToInt8(*form.PushEnabled)
	}
	if form.DndEnabled != nil {
		setting.DndEnabled = boolToInt8(*form.DndEnabled)
	}
	if form.DndStart != nil {
		if _, err := time.ParseInLocation(dndTimeFormat, *form.DndStart, time.Local); err != nil {
			return common.NewBizError(common.PARAM_ERROR, "免打扰开始时间格式不正确")
		}
		setting.DndStart = *form.DndStart
	}
	if form.DndEnd != nil {
		if _, err := time.ParseInLocation(dndTimeFormat, *form.DndEnd, time.Local); err != nil {
			return common.NewBizError(common.PARAM_ERROR, "免打扰结束时间格式不正确")
		}
		setting.DndEnd = *form.DndEnd
	}
	if form.Preferences != nil {
		var existing map[string]interface{}
		if setting.Preferences != "" {
			_ = json.Unmarshal([]byte(setting.Preferences), &existing)
		}
		if existing == nil {
			existing = make(map[string]interface{})
		}
		formPrefsBytes, _ := json.Marshal(form.Preferences)
		var formPrefsMap map[string]interface{}
		_ = json.Unmarshal(formPrefsBytes, &formPrefsMap)
		for k, v := range formPrefsMap {
			if newMap, ok := v.(map[string]interface{}); ok {
				if oldMap, ok := existing[k].(map[string]interface{}); ok {
					for k2, v2 := range newMap {
						oldMap[k2] = v2
					}
					continue
				}
			}
			existing[k] = v
		}
		setting.Preferences = toJSONString(existing)
	}

	if err := s.settingRepo.Update(ctx, setting); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新通知设置失败", err)
	}
	return nil
}

func (s *NotificationSettingService) getOrCreateDefault(ctx context.Context, userID int64) (*model.SysNotificationSetting, error) {
	setting, err := s.settingRepo.FindByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询通知设置失败", err)
	}
	if setting != nil {
		return setting, nil
	}

	defaultPrefs := vo.NotificationPreferences{
		TypeChannels: map[string]vo.TypeChannel{
			"announcement": {Push: true},
			"business":     {Push: false},
			"member":       {Push: true},
		},
		ModuleSwitches: map[string]bool{
			"prediction":   true,
			"feedback":     true,
			"announcement": true,
		},
	}
	setting = &model.SysNotificationSetting{
		UserID:      userID,
		PushEnabled: 1,
		DndEnabled:  0,
		DndStart:    "22:00:00",
		DndEnd:      "08:00:00",
		Preferences: toJSONString(defaultPrefs),
	}

	if err := s.settingRepo.Create(ctx, setting); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建默认通知设置失败", err)
	}
	return setting, nil
}

func (s *NotificationSettingService) toVO(setting *model.SysNotificationSetting) *vo.NotificationSettingsVO {
	result := &vo.NotificationSettingsVO{
		PushEnabled: setting.PushEnabled == 1,
		DndEnabled:  setting.DndEnabled == 1,
		DndStart:    setting.DndStart,
		DndEnd:      setting.DndEnd,
		Preferences: vo.NotificationPreferences{
			TypeChannels:   make(map[string]vo.TypeChannel),
			ModuleSwitches: make(map[string]bool),
		},
	}
	if setting.Preferences != "" {
		_ = json.Unmarshal([]byte(setting.Preferences), &result.Preferences)
	}
	if result.Preferences.TypeChannels == nil {
		result.Preferences.TypeChannels = make(map[string]vo.TypeChannel)
	}
	if result.Preferences.ModuleSwitches == nil {
		result.Preferences.ModuleSwitches = make(map[string]bool)
	}
	return result
}

func boolToInt8(b bool) int8 {
	if b {
		return 1
	}
	return 0
}

var _ INotificationSettingService = (*NotificationSettingService)(nil)
