package member

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type QuotaType string

const (
	QuotaTypeDehaze   QuotaType = "dehaze"
	QuotaTypeEvaluate QuotaType = "evaluate"
)

type MessageSender interface {
	Send(ctx context.Context, form *bo.MessageSendForm) (*vo.MessageSendResultVO, error)
}

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
	AwardGrowth(ctx context.Context, userID int64, changeType string, changeValue int, reason, relatedID string) error
	CheckAndDeductQuota(ctx context.Context, userID int64, quotaType QuotaType) error
	RefundQuota(ctx context.Context, userID int64, quotaType QuotaType) error
	ResetMonthlyQuota(ctx context.Context) error
	ProcessExpiredMembers(ctx context.Context) error
	SendExpireReminders(ctx context.Context) error
	GetLevelCode(ctx context.Context, userID int64) (string, error)
	GetBatchLimit(ctx context.Context, levelCode string) (int, error)
}
