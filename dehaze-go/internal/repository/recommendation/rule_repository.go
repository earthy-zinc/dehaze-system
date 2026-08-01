package recommendation

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type ruleRepository struct {
	db *gorm.DB
}

func NewRuleRepository(db *gorm.DB) RuleRepository {
	return &ruleRepository{db: db}
}

func (r *ruleRepository) FindAll(ctx context.Context) ([]model.SysRecommendationRule, error) {
	var rules []model.SysRecommendationRule
	err := r.db.WithContext(ctx).Where("deleted = 0").Order("weight ASC").Find(&rules).Error
	return rules, err
}

func (r *ruleRepository) FindEnabled(ctx context.Context) ([]model.SysRecommendationRule, error) {
	var rules []model.SysRecommendationRule
	err := r.db.WithContext(ctx).Where("deleted = 0 AND enabled = 1").Find(&rules).Error
	return rules, err
}

func (r *ruleRepository) FindByID(ctx context.Context, id int64) (*model.SysRecommendationRule, error) {
	var rule model.SysRecommendationRule
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&rule).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rule, err
}

func (r *ruleRepository) Create(ctx context.Context, rule *model.SysRecommendationRule) error {
	return r.db.WithContext(ctx).Create(rule).Error
}

func (r *ruleRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysRecommendationRule{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}
