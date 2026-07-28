package member

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type MemberSignInRepository struct {
	db *gorm.DB
}

func NewMemberSignInRepository(db *gorm.DB) *MemberSignInRepository {
	return &MemberSignInRepository{db: db}
}

func (r *MemberSignInRepository) Create(ctx context.Context, sign *model.SysMemberSignIn) error {
	return r.db.WithContext(ctx).Create(sign).Error
}

func (r *MemberSignInRepository) FindByUserIDAndDate(ctx context.Context, userID int64, date time.Time) (*model.SysMemberSignIn, error) {
	var s model.SysMemberSignIn
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND sign_date = ?", userID, date.Format("2006-01-02")).
		First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

func (r *MemberSignInRepository) FindByUserIDAndDateRange(ctx context.Context, userID int64, start, end time.Time) ([]model.SysMemberSignIn, error) {
	var list []model.SysMemberSignIn
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND sign_date >= ? AND sign_date <= ?", userID, start.Format("2006-01-02"), end.Format("2006-01-02")).
		Order("sign_date ASC").
		Find(&list).Error
	return list, err
}

func (r *MemberSignInRepository) FindLatestByUserID(ctx context.Context, userID int64) (*model.SysMemberSignIn, error) {
	var s model.SysMemberSignIn
	err := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("sign_date DESC").
		First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

var _ IMemberSignInRepository = (*MemberSignInRepository)(nil)
