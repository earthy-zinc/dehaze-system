package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type SkillRepository struct {
	db *gorm.DB
}

func NewSkillRepository(db *gorm.DB) *SkillRepository {
	return &SkillRepository{db: db}
}

// Paginate 管理员分页（含禁用，支持名称模糊与状态筛选）
func (r *SkillRepository) Paginate(ctx context.Context, page, size int, keyword string, status *int) ([]model.SysAiSkill, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiSkill{}).Where("deleted = 0")
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("name LIKE ? ESCAPE '\\\\'", pattern)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiSkill
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// ListEnabled 全部启用项（普通用户列表口径，不分页）
func (r *SkillRepository) ListEnabled(ctx context.Context) ([]model.SysAiSkill, error) {
	var items []model.SysAiSkill
	err := r.db.WithContext(ctx).
		Where("status = 1 AND deleted = 0").
		Order("id DESC").Find(&items).Error
	return items, err
}

func (r *SkillRepository) GetByID(ctx context.Context, id int64) (*model.SysAiSkill, error) {
	var s model.SysAiSkill
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

// GetByName 按名称查询；includeDeleted=true 时含软删行（name 唯一键含 deleted，查重需含历史）
func (r *SkillRepository) GetByName(ctx context.Context, name string, includeDeleted bool) (*model.SysAiSkill, error) {
	db := r.db.WithContext(ctx).Where("name = ?", name)
	if includeDeleted {
		db = db.Unscoped()
	} else {
		db = db.Where("deleted = 0")
	}
	var s model.SysAiSkill
	err := db.First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

func (r *SkillRepository) Create(ctx context.Context, s *model.SysAiSkill) error {
	return r.db.WithContext(ctx).Create(s).Error
}

func (r *SkillRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiSkill{}).Where("id = ?", id).Updates(updates).Error
}

func (r *SkillRepository) SoftDelete(ctx context.Context, id, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiSkill{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// ListMarket 市场目录：已共享且启用的 Skill
func (r *SkillRepository) ListMarket(ctx context.Context) ([]model.SysAiSkill, error) {
	var items []model.SysAiSkill
	err := r.db.WithContext(ctx).
		Where("market_shared = 1 AND status = 1 AND deleted = 0").
		Order("id ASC").Find(&items).Error
	return items, err
}

// CountByNames 批量统计各 Skill 被 Agent 关联数（键为 skill name）
func (r *SkillRepository) CountByNames(ctx context.Context, names []string) (map[string]int64, error) {
	result := make(map[string]int64, len(names))
	if len(names) == 0 {
		return result, nil
	}
	var rows []struct {
		SkillName string `gorm:"column:skill_name"`
		Cnt       int64  `gorm:"column:cnt"`
	}
	err := r.db.WithContext(ctx).Table("sys_ai_agent_skill").
		Select("skill_name, COUNT(*) AS cnt").
		Where("skill_name IN ?", names).
		Group("skill_name").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		result[row.SkillName] = row.Cnt
	}
	return result, nil
}

// ==================== SKILL 目录文件清单 ====================

func (r *SkillRepository) ListFiles(ctx context.Context, skillID int64) ([]model.SysAiSkillFile, error) {
	var items []model.SysAiSkillFile
	err := r.db.WithContext(ctx).
		Where("skill_id = ?", skillID).
		Order("path ASC").Find(&items).Error
	return items, err
}

func (r *SkillRepository) CreateFiles(ctx context.Context, files []model.SysAiSkillFile) error {
	if len(files) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Create(&files).Error
}

func (r *SkillRepository) DeleteFiles(ctx context.Context, skillID int64) error {
	return r.db.WithContext(ctx).Where("skill_id = ?", skillID).Delete(&model.SysAiSkillFile{}).Error
}
