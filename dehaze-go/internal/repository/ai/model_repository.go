package ai

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// Repository AI 域仓储集合（A 类管理面与 B 类代理共用同一批数据访问方法）
type ModelRepository struct {
	db *gorm.DB
}

func NewModelRepository(db *gorm.DB) *ModelRepository {
	return &ModelRepository{db: db}
}

// PaginateModels 分页查询模型（keyword 匹配 display_name/model_id，model_type 精确筛选）
func (r *ModelRepository) PaginateModels(ctx context.Context, page, size int, keyword, modelType string) ([]model.SysAiModel, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiModel{}).Where("deleted = 0")
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("(display_name LIKE ? ESCAPE '\\\\' OR model_id LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	if modelType != "" {
		db = db.Where("model_type = ?", modelType)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiModel
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// GetByPK 按主键查询未删除模型
func (r *ModelRepository) GetByPK(ctx context.Context, id int64) (*model.SysAiModel, error) {
	var m model.SysAiModel
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&m).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &m, err
}

// GetByModelID 按业务键 model_id 查询（同 model_id 多供应商时取 id 最小行）
func (r *ModelRepository) GetByModelID(ctx context.Context, modelID string) (*model.SysAiModel, error) {
	var m model.SysAiModel
	err := r.db.WithContext(ctx).Where("model_id = ? AND deleted = 0", modelID).Order("id ASC").First(&m).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &m, err
}

// GetByModelAndProvider 按 (model_id, provider_id) 联合键查询
func (r *ModelRepository) GetByModelAndProvider(ctx context.Context, modelID string, providerID int64) (*model.SysAiModel, error) {
	var m model.SysAiModel
	err := r.db.WithContext(ctx).
		Where("model_id = ? AND provider_id = ? AND deleted = 0", modelID, providerID).
		First(&m).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &m, err
}

// ListEnabled 查询启用模型（按 provider_id, display_name 排序，与 python list_enabled 一致）
func (r *ModelRepository) ListEnabled(ctx context.Context, modelType string) ([]model.SysAiModel, error) {
	db := r.db.WithContext(ctx).Where("status = 1 AND deleted = 0")
	if modelType != "" {
		db = db.Where("model_type = ?", modelType)
	}
	var items []model.SysAiModel
	err := db.Order("provider_id ASC, display_name ASC").Find(&items).Error
	return items, err
}

// ExistsEnabledByModelID 判断某 model_id 是否存在启用行（对齐 python `list_enabled_by_model_id` 的非空判定）
func (r *ModelRepository) ExistsEnabledByModelID(ctx context.Context, modelID string) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiModel{}).
		Where("model_id = ? AND status = 1 AND deleted = 0", modelID).
		Count(&count).Error
	return count > 0, err
}

// ListEnabledByPKs 按主键查询启用模型（降级链候选）
func (r *ModelRepository) ListEnabledByPKs(ctx context.Context, pks []int64) ([]model.SysAiModel, error) {
	if len(pks) == 0 {
		return []model.SysAiModel{}, nil
	}
	var items []model.SysAiModel
	err := r.db.WithContext(ctx).
		Where("id IN ? AND status = 1 AND deleted = 0", pks).
		Order("id ASC").Find(&items).Error
	return items, err
}

// CountFallbackTargets 统计把该模型作为降级目标的启用模型数
func (r *ModelRepository) CountFallbackTargets(ctx context.Context, pk int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiModel{}).
		Where("fallback_model_id = ? AND status = 1 AND deleted = 0", pk).
		Count(&count).Error
	return count, err
}

// CountActiveConversations 统计仍在使用该模型的活跃会话数
func (r *ModelRepository) CountActiveConversations(ctx context.Context, modelID string) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Table("sys_ai_conversation").
		Where("model = ? AND deleted = 0 AND status = 1", modelID).
		Count(&count).Error
	return count, err
}

// ListActiveConversationUsers 查询使用该模型的活跃会话用户（去重，供下线通知）
func (r *ModelRepository) ListActiveConversationUsers(ctx context.Context, modelID string) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).Table("sys_ai_conversation").
		Distinct("user_id").
		Where("model = ? AND deleted = 0 AND status = 1", modelID).
		Pluck("user_id", &ids).Error
	return ids, err
}

func (r *ModelRepository) Create(ctx context.Context, m *model.SysAiModel) error {
	return r.db.WithContext(ctx).Create(m).Error
}

func (r *ModelRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiModel{}).Where("id = ?", id).Updates(updates).Error
}

func (r *ModelRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiModel{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// escapeLike 转义 LIKE 通配符，避免用户输入中的 % / _ 被当作通配符
func escapeLike(s string) string {
	return strings.NewReplacer("\\", "\\\\", "%", "\\%", "_", "\\_").Replace(s)
}
