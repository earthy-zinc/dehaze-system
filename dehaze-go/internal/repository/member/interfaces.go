package member

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type IMemberRepository interface {
	FindByUserID(ctx context.Context, userID int64) (*model.SysMember, error)
	FindWithUserByUserID(ctx context.Context, userID int64) (*MemberWithUser, error)
	FindPageWithUser(ctx context.Context, q *query.MemberPageQuery) ([]MemberWithUser, int64, error)
	FindAllActive(ctx context.Context, excludeQuotaResetMonth *int, limit int) ([]model.SysMember, error)
	FindExpiredNonGrowth(ctx context.Context, now time.Time) ([]model.SysMember, error)
	FindExpiringBetween(ctx context.Context, start, end time.Time) ([]model.SysMember, error)
	FindUserIDsByLevelCodes(ctx context.Context, levelCodes []string) ([]int64, error)
	Upsert(ctx context.Context, m *model.SysMember) error
	UpdateLevel(ctx context.Context, userID int64, updates map[string]interface{}) error
	UpdateGrowth(ctx context.Context, userID int64, growthValue int64) error
	Update(ctx context.Context, userID int64, updates map[string]interface{}) error
	IncrementQuotaUsed(ctx context.Context, userID int64, quotaType string, delta int) error
	ResetMonthlyQuota(ctx context.Context, userID int64, dehazeQuota, evaluateQuota, quotaMonth int) error
	CreateQuotaArchive(ctx context.Context, quota *model.SysMemberQuota) error
	Transaction(ctx context.Context, fn func(repo IMemberRepository) error) error
}

type IMemberBenefitRepository interface {
	FindAll(ctx context.Context) ([]model.SysMemberBenefit, error)
	FindByLevelCode(ctx context.Context, levelCode string) (*model.SysMemberBenefit, error)
	Update(ctx context.Context, levelCode string, updates map[string]interface{}) error
	ExistsByLevelCode(ctx context.Context, levelCode string, excludeID ...int64) (bool, error)
}

type IMemberGrowthLogRepository interface {
	Create(ctx context.Context, log *model.SysMemberGrowthLog) error
	FindPageByUserID(ctx context.Context, userID int64, q *query.GrowthLogQuery) ([]model.SysMemberGrowthLog, int64, error)
}

type IMemberSignInRepository interface {
	Create(ctx context.Context, sign *model.SysMemberSignIn) error
	FindByUserIDAndDate(ctx context.Context, userID int64, date time.Time) (*model.SysMemberSignIn, error)
	FindByUserIDAndDateRange(ctx context.Context, userID int64, start, end time.Time) ([]model.SysMemberSignIn, error)
	FindLatestByUserID(ctx context.Context, userID int64) (*model.SysMemberSignIn, error)
}

type MemberWithUser struct {
	model.SysMember
	Username string `gorm:"column:username"`
	Nickname string `gorm:"column:nickname"`
	Avatar   string `gorm:"column:avatar"`
}
