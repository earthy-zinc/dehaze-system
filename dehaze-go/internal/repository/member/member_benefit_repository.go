package member

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type MemberBenefitRepository struct {
	db *gorm.DB
}

func NewMemberBenefitRepository(db *gorm.DB) *MemberBenefitRepository {
	return &MemberBenefitRepository{db: db}
}

func (r *MemberBenefitRepository) FindAll(ctx context.Context) ([]model.SysMemberBenefit, error) {
	var list []model.SysMemberBenefit
	err := r.db.WithContext(ctx).
		Where("deleted = 0").
		Order("sort ASC").
		Find(&list).Error
	return list, err
}

func (r *MemberBenefitRepository) FindByLevelCode(ctx context.Context, levelCode string) (*model.SysMemberBenefit, error) {
	var b model.SysMemberBenefit
	err := r.db.WithContext(ctx).
		Where("level_code = ? AND deleted = 0", levelCode).
		First(&b).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &b, err
}

func (r *MemberBenefitRepository) Update(ctx context.Context, levelCode string, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMemberBenefit{}).
		Where("level_code = ? AND deleted = 0", levelCode).
		Updates(updates).Error
}

var _ IMemberBenefitRepository = (*MemberBenefitRepository)(nil)
