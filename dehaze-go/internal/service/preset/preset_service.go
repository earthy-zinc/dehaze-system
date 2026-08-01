package preset

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model"
	presetrepo "github.com/earthyzinc/dehaze-go/internal/repository/preset"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const defaultCustomPresetLimit = 3

// 系统预设种子数据（对齐 Python seed_system_presets）
var systemPresetSeeds = []struct {
	Name        string
	AlgorithmID int64
	Params      json.RawMessage
	IsDefault   int8
}{
	{Name: "默认去雾", AlgorithmID: 13, Params: json.RawMessage(`{"gamma":1.0,"strength":"medium"}`), IsDefault: 1},
	{Name: "轻度去雾", AlgorithmID: 13, Params: json.RawMessage(`{"gamma":0.8,"strength":"light"}`), IsDefault: 0},
	{Name: "深度去雾", AlgorithmID: 13, Params: json.RawMessage(`{"gamma":1.5,"strength":"strong"}`), IsDefault: 0},
}

type PresetService struct {
	db        *gorm.DB
	repo      presetrepo.IPresetRepository
	memberSvc memberservice.IMemberService
}

func NewPresetService(db *gorm.DB, repo presetrepo.IPresetRepository, memberSvc memberservice.IMemberService) *PresetService {
	return &PresetService{db: db, repo: repo, memberSvc: memberSvc}
}

// PresetVO 参数预设视图
type PresetVO struct {
	ID          int64           `json:"id"`
	Name        string          `json:"name"`
	Type        string          `json:"type"`
	AlgorithmID int64           `json:"algorithmId"`
	Params      json.RawMessage `json:"params"`
	UserID      *int64          `json:"userId"`
	IsDefault   int8            `json:"isDefault"`
	CreateTime  string          `json:"createTime"`
}

func toPresetVO(p *model.SysPreset) PresetVO {
	vo := PresetVO{
		ID:          p.ID,
		Name:        p.Name,
		Type:        p.Type,
		AlgorithmID: p.AlgorithmID,
		IsDefault:   p.IsDefault,
		UserID:      p.UserID,
		CreateTime:  p.CreatedAt.Format("2006-01-02 15:04:05"),
	}
	if p.Params != nil {
		vo.Params = *p.Params
	}
	return vo
}

// ListPresets 获取预设列表（系统预设 + 用户自定义）
func (s *PresetService) ListPresets(ctx context.Context, algorithmID int64, userID int64, isSystem *bool, page, pageSize int) (*common.PageResult, error) {
	if page < 1 {
		page = 1
	}
	if pageSize < 1 || pageSize > 100 {
		pageSize = 10
	}

	list, total, err := s.repo.FindPage(ctx, algorithmID, userID, isSystem, page, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预设列表失败", err)
	}
	result := make([]PresetVO, 0, len(list))
	for i := range list {
		result = append(result, toPresetVO(&list[i]))
	}
	return &common.PageResult{
		List:     result,
		Total:    total,
		Page:     page,
		PageSize: pageSize,
	}, nil
}

// PresetForm 预设表单
type PresetForm struct {
	Name        string          `json:"name" binding:"required"`
	AlgorithmID int64           `json:"algorithmId" binding:"required"`
	Params      json.RawMessage `json:"params" binding:"required"`
	IsDefault   *int8           `json:"isDefault"`
}

// CreatePreset 创建自定义预设
func (s *PresetService) CreatePreset(ctx context.Context, userID int64, form *PresetForm) (*PresetVO, error) {
	levelCode, _ := s.memberSvc.GetLevelCode(ctx, userID)
	limit := s.getPresetLimitByLevel(levelCode)

	count, err := s.repo.CountByUser(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预设数量失败", err)
	}
	if limit > 0 && int(count) >= limit {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "自定义预设数量已达上限")
	}

	isDefault := int8(0)
	if form.IsDefault != nil {
		isDefault = *form.IsDefault
	}

	preset := &model.SysPreset{
		BaseModel:   model.BaseModel{CreateBy: userID},
		Name:        form.Name,
		Type:        "custom",
		AlgorithmID: form.AlgorithmID,
		Params:      &form.Params,
		UserID:      &userID,
		IsDefault:   isDefault,
	}
	if err := s.repo.Create(ctx, preset); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建预设失败", err)
	}
	vo := toPresetVO(preset)
	return &vo, nil
}

// UpdatePreset 更新自定义预设
func (s *PresetService) UpdatePreset(ctx context.Context, id int64, userID int64, form *PresetForm) (*PresetVO, error) {
	preset, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "预设不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询预设失败", err)
	}

	if preset.Type == "system" {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "系统预设不可修改")
	}
	if preset.UserID == nil || *preset.UserID != userID {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "只能操作自己的预设")
	}

	updates := map[string]interface{}{
		"name":         form.Name,
		"algorithm_id": form.AlgorithmID,
		"params":       form.Params,
		"update_by":    userID,
	}
	if form.IsDefault != nil {
		updates["is_default"] = *form.IsDefault
	}
	if err := s.repo.Update(ctx, id, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新预设失败", err)
	}

	updated, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询更新后的预设失败", err)
	}
	vo := toPresetVO(updated)
	return &vo, nil
}

// DeletePreset 删除自定义预设
func (s *PresetService) DeletePreset(ctx context.Context, id int64, userID int64) error {
	preset, err := s.repo.FindByID(ctx, id)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "预设不存在")
		}
		return common.WrapBizError(common.DATABASE_ERROR, "查询预设失败", err)
	}

	if preset.Type == "system" {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "系统预设不可删除")
	}
	if preset.UserID == nil || *preset.UserID != userID {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "只能操作自己的预设")
	}

	return s.repo.Delete(ctx, id)
}

func (s *PresetService) getPresetLimitByLevel(levelCode string) int {
	// 自定义预设数量：普通3/VIP1-10/VIP2-20/SVIP-无限制(返回0)
	switch levelCode {
	case "level_0":
		return 3
	case "level_1":
		return 10
	case "level_2":
		return 20
	case "level_3":
		return 0 // SVIP 无限制
	}
	return defaultCustomPresetLimit
}

// SeedSystemPresets 初始化系统预设种子数据（幂等：已有数据则跳过）
func SeedSystemPresets(db *gorm.DB) {
	ctx := context.Background()
	var count int64
	if err := db.WithContext(ctx).Model(&model.SysPreset{}).Where("type = ?", "system").Count(&count).Error; err != nil {
		logger.Error("查询系统预设数量失败", zap.Error(err))
		return
	}
	if count > 0 {
		return
	}

	for _, seed := range systemPresetSeeds {
		preset := &model.SysPreset{
			Name:        seed.Name,
			Type:        "system",
			AlgorithmID: seed.AlgorithmID,
			Params:      &seed.Params,
			IsDefault:   seed.IsDefault,
		}
		if err := db.WithContext(ctx).Create(preset).Error; err != nil {
			logger.Error("创建系统预设种子数据失败", zap.String("name", seed.Name), zap.Error(err))
		}
	}
	logger.Info("系统预设种子数据初始化完成", zap.Int("count", len(systemPresetSeeds)))
}
