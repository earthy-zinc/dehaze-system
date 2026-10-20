package vo

type PackagePageVO struct {
	ID            int64  `json:"id"`
	Name          string `json:"name"`
	PackageType   string `json:"packageType"`
	LevelCode     string `json:"levelCode"`
	LevelName     string `json:"levelName"`
	Period        string `json:"period"`
	PeriodDays    int    `json:"periodDays"`
	OriginalPrice int64  `json:"originalPrice"`
	SalePrice     int64  `json:"salePrice"`
	DailyPrice    int64  `json:"dailyPrice"`
	SalesCount    int64  `json:"salesCount"`
	Status        int    `json:"status"`
	CreateTime    string `json:"createTime"`
}

type PackageDetailVO struct {
	ID            int64  `json:"id"`
	Name          string `json:"name"`
	PackageType   string `json:"packageType"`
	LevelCode     string `json:"levelCode"`
	LevelName     string `json:"levelName"`
	Period        string `json:"period"`
	PeriodDays    int    `json:"periodDays"`
	OriginalPrice int64  `json:"originalPrice"`
	SalePrice     int64  `json:"salePrice"`
	DailyPrice    int64  `json:"dailyPrice"`
	CreditAmount  *int64 `json:"creditAmount"`
	// 积分卡单价 = salePrice / creditAmount（会员卡为 0）
	CreditUnitPrice int64          `json:"creditUnitPrice"`
	Description     string         `json:"description"`
	Benefits        map[string]int `json:"benefits"`
	SalesCount      int64          `json:"salesCount"`
	// 进行中促销活动仅详情返回（python get_detail 同口径），在售列表必须省略该属性
	ActivePromotions []PromotionVO `json:"activePromotions,omitempty"`
}

type PromotionVO struct {
	ID            int64                  `json:"id"`
	Name          string                 `json:"name"`
	Type          string                 `json:"type"`
	Description   string                 `json:"description"`
	StartTime     string                 `json:"startTime"`
	EndTime       string                 `json:"endTime"`
	ActivityRules map[string]interface{} `json:"activityRules"`
	NewUserOnly   int                    `json:"newUserOnly"`
	Status        int                    `json:"status"`
	CreateTime    string                 `json:"createTime,omitempty"`
}

type PriceResult struct {
	OriginalPrice  int64 `json:"originalPrice"`
	DiscountAmount int64 `json:"discountAmount"`
	CouponAmount   int64 `json:"couponAmount"`
	PayableAmount  int64 `json:"payableAmount"`
}

type SalesStatsVO struct {
	TotalSales   int64                  `json:"totalSales"`
	TotalRevenue int64                  `json:"totalRevenue"`
	PackageStats []PackageSalesStatItem `json:"packageStats"`
	LevelStats   []LevelSalesStatItem   `json:"levelStats"`
	PeriodStats  []PeriodSalesStatItem  `json:"periodStats"`
	CouponStats  CouponStatsVO          `json:"couponStats"`
}

type PackageSalesStatItem struct {
	PackageID   int64  `json:"packageId"`
	PackageName string `json:"packageName"`
	SalesCount  int64  `json:"salesCount"`
	Revenue     int64  `json:"revenue"`
}

type LevelSalesStatItem struct {
	LevelCode  string `json:"levelCode"`
	LevelName  string `json:"levelName"`
	SalesCount int64  `json:"salesCount"`
	Revenue    int64  `json:"revenue"`
}

type PeriodSalesStatItem struct {
	Period     string `json:"period"`
	PeriodName string `json:"periodName"`
	SalesCount int64  `json:"salesCount"`
	Revenue    int64  `json:"revenue"`
}

type CouponStatsVO struct {
	TotalIssued int64   `json:"totalIssued"`
	TotalUsed   int64   `json:"totalUsed"`
	UsageRate   float64 `json:"usageRate"`
}
