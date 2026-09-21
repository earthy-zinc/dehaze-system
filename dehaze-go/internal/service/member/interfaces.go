package member

import (
	"context"
	"time"

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

// TrialStatusDeps 试用引导状态所需的外部取数。
// 只声明 member 自身需要的最小能力（窄接口），由 coupon / AI 积分流水 / order 模块在组装层提供
// 适配器，member 包不引用它们的结构体，保持自洽。
type TrialStatusDeps interface {
	// ActiveTrialCouponExpireTime 有效体验券（未使用、未过期）的最晚到期时间；无券返回 nil, nil
	ActiveTrialCouponExpireTime(ctx context.Context, userID int64) (*time.Time, error)
	// SumTrialCredits 按 source='trial' 汇总 AI 积分（无记录返回 0）
	SumTrialCredits(ctx context.Context, userID int64) (int64, error)
	// HasPaidOrder 是否存在已支付订单（status 2 已支付 / 3 已完成）
	HasPaidOrder(ctx context.Context, userID int64) (bool, error)
}

// MemberAuditLog 审计日志条目（member 包自洽的取数结构，不依赖审计模块的模型/bson 标签）
type MemberAuditLog struct {
	ID          string
	OperatorID  int64
	Action      string
	Module      string
	BeforeValue any
	AfterValue  any
	IP          string
	CreateTime  time.Time
}

// AICreditsProvider AI 积分类目取数（余额 + 当日已用），由 AI 计费模块在组装层适配
// （python：balance_service.get_balance + quota_service.get_used）
type AICreditsProvider interface {
	AICredits(ctx context.Context, userID int64) (balance, todayUsed int64, err error)
}

// PackageOverridesProvider 已购会员卡的权益覆盖项（python package_repository.get_by_level_code
// 取 package.benefit_overrides，key 为 camelCase），由套餐模块在组装层适配。
type PackageOverridesProvider interface {
	PackageBenefitOverrides(ctx context.Context, levelCode string) (map[string]int, error)
}

// AuditLogLister 会员操作日志取数（窄接口，由审计日志仓储在组装层适配）
type AuditLogLister interface {
	ListByTarget(ctx context.Context, targetType string, targetID int64, page, pageSize int) ([]MemberAuditLog, int64, error)
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
	GetMaxDevices(ctx context.Context, userID int64) (int, error)
	GetBatchLimit(ctx context.Context, levelCode string) (int, error)
	InitDefaultMember(ctx context.Context, userID int64) error
	EnsureMemberProfile(ctx context.Context, userID int64) error
	GetBenefitSummary(ctx context.Context, userID int64) (*vo.MemberBenefitSummaryVO, error)
	GetTrialStatus(ctx context.Context, userID int64) (*vo.MemberTrialStatusVO, error)
	ListMemberAuditLogs(ctx context.Context, userID int64, page, pageSize int) (*vo.MemberAuditLogPageVO, error)
}
