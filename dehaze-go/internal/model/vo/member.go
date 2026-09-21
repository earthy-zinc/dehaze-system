package vo

type BenefitVO struct {
	LevelCode            string `json:"levelCode"`
	LevelName            string `json:"levelName"`
	GrowthMin            int64  `json:"growthMin"`
	GrowthMax            int64  `json:"growthMax"`
	MonthlyDehazeQuota   int    `json:"monthlyDehazeQuota"`
	MonthlyEvaluateQuota int    `json:"monthlyEvaluateQuota"`
	HistoryRetention     int    `json:"historyRetention"`
	BatchLimit           int    `json:"batchLimit"`
	MaxDevices           int    `json:"maxDevices"`
	Priority             int    `json:"priority"`
	AdvancedParams       int    `json:"advancedParams"`
	HdExport             int    `json:"hdExport"`
	ReportExport         int    `json:"reportExport"`
	BatchDownload        int    `json:"batchDownload"`
	Sort                 int    `json:"sort"`
	Status               int    `json:"status"`
}

type MemberProfileVO struct {
	UserID               int64     `json:"userId"`
	Username             string    `json:"username"`
	Nickname             string    `json:"nickname"`
	Avatar               string    `json:"avatar"`
	LevelCode            string    `json:"levelCode"`
	LevelSource          string    `json:"levelSource"`
	LevelName            string    `json:"levelName"`
	GrowthValue          int64     `json:"growthValue"`
	NextLevelGrowth      int64     `json:"nextLevelGrowth"`
	ProgressPercent      int       `json:"progressPercent"`
	ExpireTime           string    `json:"expireTime"`
	MonthlyDehazeQuota   int       `json:"monthlyDehazeQuota"`
	MonthlyDehazeUsed    int       `json:"monthlyDehazeUsed"`
	MonthlyEvaluateQuota int       `json:"monthlyEvaluateQuota"`
	MonthlyEvaluateUsed  int       `json:"monthlyEvaluateUsed"`
	Benefits             BenefitVO `json:"benefits"`
	Status               int       `json:"status"`
}

// MemberBenefitSummaryVO 会员权益概览（对齐 python `get_benefit_summary` 的 payload）。
// 用户端 `/members/benefit-summary` 与管理端 `/members/{id}/benefit-usage` 共用同一结构。
type MemberBenefitSummaryVO struct {
	ImageCategory    BenefitImageCategoryVO    `json:"imageCategory"`
	EvaluateCategory BenefitEvaluateCategoryVO `json:"evaluateCategory"`
	AICategory       BenefitAICategoryVO       `json:"aiCategory"`
}

// BenefitAICategoryVO AI 积分类目：余额/今日已用/日限额/月限额。
// 限额取"等级权益与已购会员卡覆盖值的较高值"（python get_benefit_summary）。
type BenefitAICategoryVO struct {
	CreditsBalance int `json:"creditsBalance"`
	TodayUsed      int `json:"todayUsed"`
	DailyLimit     int `json:"dailyLimit"`
	MonthlyLimit   int `json:"monthlyLimit"`
}

// MemberAuditLogPageVO 会员操作日志分页（python `list_member_audit_logs` 只回 list/total）。
type MemberAuditLogPageVO struct {
	List  []MemberAuditLogVO `json:"list"`
	Total int64              `json:"total"`
}

// MemberAuditLogVO 单条会员操作日志（字段名对齐 python 映射的 camelCase）。
type MemberAuditLogVO struct {
	ID          string `json:"id"`
	OperatorID  int64  `json:"operatorId"`
	Action      string `json:"action"`
	Module      string `json:"module"`
	BeforeValue any    `json:"beforeValue"`
	AfterValue  any    `json:"afterValue"`
	IP          string `json:"ip"`
	CreateTime  string `json:"createTime"`
}

// MemberTrialStatusVO 试用引导状态（对齐 python `get_trial_status` 的 payload）。
type MemberTrialStatusVO struct {
	ShowTrialEntry            bool    `json:"showTrialEntry"`
	TrialDays                 int     `json:"trialDays"`
	TrialCredits              int     `json:"trialCredits"`
	VoucherActivated          bool    `json:"voucherActivated"`
	VoucherExpireTime         *string `json:"voucherExpireTime"`
	AITrialCreditsBalance     int64   `json:"aiTrialCreditsBalance"`
	NewUserExclusiveAvailable bool    `json:"newUserExclusiveAvailable"`
	PaidMembership            bool    `json:"paidMembership"`
}

// BenefitImageCategoryVO 图像处理类目：remaining 为各任务剩余次数的最低值，details 为逐任务明细。
type BenefitImageCategoryVO struct {
	Remaining int                  `json:"remaining"`
	Details   []BenefitImageTaskVO `json:"details"`
}

type BenefitImageTaskVO struct {
	TaskType  string `json:"taskType"`
	Quota     int    `json:"quota"`
	Used      int    `json:"used"`
	Remaining int    `json:"remaining"`
}

type BenefitEvaluateCategoryVO struct {
	Remaining int `json:"remaining"`
}

type MemberPageVO struct {
	UserID           int64  `json:"userId"`
	Username         string `json:"username"`
	Nickname         string `json:"nickname"`
	LevelCode        string `json:"levelCode"`
	LevelName        string `json:"levelName"`
	GrowthValue      int64  `json:"growthValue"`
	MonthlyUsed      int    `json:"monthlyUsed"`
	ExpireTime       string `json:"expireTime"`
	Status           int    `json:"status"`
	BecomeMemberTime string `json:"becomeMemberTime"`
}

type MemberDetailVO struct {
	MemberProfileVO
	LevelSource      string `json:"levelSource"`
	TotalConsumption int64  `json:"totalConsumption"`
	BecomeMemberTime string `json:"becomeMemberTime"`
	FrozenReason     string `json:"frozenReason"`
	FrozenTime       string `json:"frozenTime"`
	QuotaResetMonth  int    `json:"quotaResetMonth"`
}

type GrowthLogVO struct {
	ID          int64  `json:"id"`
	ChangeType  string `json:"changeType"`
	ChangeValue int    `json:"changeValue"`
	Balance     int64  `json:"balance"`
	RelatedID   string `json:"relatedId"`
	Reason      string `json:"reason"`
	OperatorID  *int64 `json:"operatorId"`
	CreateTime  string `json:"createTime"`
}

type SignInResultVO struct {
	SignDate       string `json:"signDate"`
	ContinuousDays int    `json:"continuousDays"`
	GrowthValue    int    `json:"growthValue"`
	BonusGrowth    int    `json:"bonusGrowth"`
}

type SignInCalendarVO struct {
	SignDates      []string `json:"signDates"`
	ContinuousDays int      `json:"continuousDays"`
	TotalDays      int      `json:"totalDays"`
}
