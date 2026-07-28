package member

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IMemberService interface {
	GetProfile(ctx context.Context, userID int64) (*vo.MemberProfileVO, error)
	ListGrowthLogs(ctx context.Context, userID int64, q *query.GrowthLogQuery) (*vo.PageResult[vo.GrowthLogVO], error)
	SignIn(ctx context.Context, userID int64) (*vo.SignInResultVO, error)
	GetSignInCalendar(ctx context.Context, userID int64, year, month int) (*vo.SignInCalendarVO, error)
	ListPagedMembers(ctx context.Context, q *query.MemberPageQuery) (*vo.PageResult[vo.MemberPageVO], error)
	GetMemberDetail(ctx context.Context, userID int64) (*vo.MemberDetailVO, error)
	AdjustLevel(ctx context.Context, userID, operatorID int64, form *bo.MemberLevelAdjustForm) error
	AdjustGrowth(ctx context.Context, userID, operatorID int64, form *bo.MemberGrowthAdjustForm) error
	UpdateStatus(ctx context.Context, userID int64, form *bo.MemberStatusForm) error
	ListBenefits(ctx context.Context) ([]vo.BenefitVO, error)
	UpdateBenefit(ctx context.Context, levelCode string, form *bo.BenefitForm) error
}
