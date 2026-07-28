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
	Priority             int    `json:"priority"`
	AdvancedParams       int    `json:"advancedParams"`
	HdExport             int    `json:"hdExport"`
	ReportExport         int    `json:"reportExport"`
	BatchDownload        int    `json:"batchDownload"`
	Sort                 int    `json:"sort"`
	Status               int    `json:"status"`
}

type MemberProfileVO struct {
	UserID                int64      `json:"userId"`
	Username              string     `json:"username"`
	Nickname              string     `json:"nickname"`
	Avatar                string     `json:"avatar"`
	LevelCode             string     `json:"levelCode"`
	LevelName             string     `json:"levelName"`
	GrowthValue           int64      `json:"growthValue"`
	NextLevelGrowth       int64      `json:"nextLevelGrowth"`
	ProgressPercent       int        `json:"progressPercent"`
	ExpireTime            string     `json:"expireTime"`
	MonthlyDehazeQuota    int        `json:"monthlyDehazeQuota"`
	MonthlyDehazeUsed     int        `json:"monthlyDehazeUsed"`
	MonthlyEvaluateQuota  int        `json:"monthlyEvaluateQuota"`
	MonthlyEvaluateUsed   int        `json:"monthlyEvaluateUsed"`
	Benefits              BenefitVO  `json:"benefits"`
	Status                int        `json:"status"`
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
