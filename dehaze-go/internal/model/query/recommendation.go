package query

// RecommendationReportQuery 推荐报表查询参数
type RecommendationReportQuery struct {
	StartDate string `json:"startDate" form:"startDate"`
	EndDate   string `json:"endDate" form:"endDate"`
}
